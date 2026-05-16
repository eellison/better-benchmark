"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s3_g26
Pattern hash: f9a16eb02e03
Shape hash: 8d37a434
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg263_1: "f16[128, 1000]", mm_1: "f16[1000, 192]", select_scatter: "f32[128, 197, 192]", arg188_1: "f32[128, 197, 192]", view_1: "f16[25216, 192]", mm_3: "f16[192, 768]", view_5: "f16[25216, 768]", mm_5: "f16[768, 192]", convert_element_type_9: "f32[128, 197, 192]", arg184_1: "f32[128, 197, 192]", view_8: "f16[25216, 192]", mm_7: "f16[192, 192]", view_16: "f16[25216, 576]", mm_9: "f16[576, 192]", convert_element_type_15: "f32[128, 197, 192]", arg175_1: "f32[128, 197, 192]", view_19: "f16[25216, 192]", mm_11: "f16[192, 768]", view_23: "f16[25216, 768]", mm_13: "f16[768, 192]", convert_element_type_24: "f32[128, 197, 192]", arg171_1: "f32[128, 197, 192]", view_26: "f16[25216, 192]", mm_15: "f16[192, 192]", view_34: "f16[25216, 576]", mm_17: "f16[576, 192]", convert_element_type_30: "f32[128, 197, 192]", arg162_1: "f32[128, 197, 192]", view_37: "f16[25216, 192]", mm_19: "f16[192, 768]", view_41: "f16[25216, 768]", mm_21: "f16[768, 192]", convert_element_type_39: "f32[128, 197, 192]", arg158_1: "f32[128, 197, 192]", view_44: "f16[25216, 192]", mm_23: "f16[192, 192]", view_52: "f16[25216, 576]", mm_25: "f16[576, 192]", convert_element_type_45: "f32[128, 197, 192]", arg149_1: "f32[128, 197, 192]", view_55: "f16[25216, 192]", mm_27: "f16[192, 768]", view_59: "f16[25216, 768]", mm_29: "f16[768, 192]", convert_element_type_54: "f32[128, 197, 192]", arg145_1: "f32[128, 197, 192]", view_62: "f16[25216, 192]", mm_31: "f16[192, 192]", view_70: "f16[25216, 576]", mm_33: "f16[576, 192]", convert_element_type_60: "f32[128, 197, 192]", arg136_1: "f32[128, 197, 192]", view_73: "f16[25216, 192]", mm_35: "f16[192, 768]", view_77: "f16[25216, 768]", mm_37: "f16[768, 192]", convert_element_type_69: "f32[128, 197, 192]", arg132_1: "f32[128, 197, 192]", view_80: "f16[25216, 192]", mm_39: "f16[192, 192]", view_88: "f16[25216, 576]", mm_41: "f16[576, 192]", convert_element_type_75: "f32[128, 197, 192]", arg123_1: "f32[128, 197, 192]", view_91: "f16[25216, 192]", mm_43: "f16[192, 768]", view_95: "f16[25216, 768]", mm_45: "f16[768, 192]", convert_element_type_84: "f32[128, 197, 192]", arg119_1: "f32[128, 197, 192]", view_98: "f16[25216, 192]", mm_47: "f16[192, 192]", view_106: "f16[25216, 576]", mm_49: "f16[576, 192]", convert_element_type_90: "f32[128, 197, 192]", arg110_1: "f32[128, 197, 192]", view_109: "f16[25216, 192]", mm_51: "f16[192, 768]", view_113: "f16[25216, 768]", mm_53: "f16[768, 192]", convert_element_type_99: "f32[128, 197, 192]", arg106_1: "f32[128, 197, 192]", view_116: "f16[25216, 192]", mm_55: "f16[192, 192]", view_124: "f16[25216, 576]", mm_57: "f16[576, 192]", convert_element_type_105: "f32[128, 197, 192]", arg97_1: "f32[128, 197, 192]", view_127: "f16[25216, 192]", mm_59: "f16[192, 768]", view_131: "f16[25216, 768]", mm_61: "f16[768, 192]", convert_element_type_114: "f32[128, 197, 192]", arg93_1: "f32[128, 197, 192]", view_134: "f16[25216, 192]", mm_63: "f16[192, 192]", view_142: "f16[25216, 576]", mm_65: "f16[576, 192]", convert_element_type_120: "f32[128, 197, 192]", arg84_1: "f32[128, 197, 192]", view_145: "f16[25216, 192]", mm_67: "f16[192, 768]", view_149: "f16[25216, 768]", mm_69: "f16[768, 192]", convert_element_type_129: "f32[128, 197, 192]", arg80_1: "f32[128, 197, 192]", view_152: "f16[25216, 192]", mm_71: "f16[192, 192]", view_160: "f16[25216, 576]", mm_73: "f16[576, 192]", convert_element_type_135: "f32[128, 197, 192]", arg71_1: "f32[128, 197, 192]", view_163: "f16[25216, 192]", mm_75: "f16[192, 768]", view_167: "f16[25216, 768]", mm_77: "f16[768, 192]", convert_element_type_144: "f32[128, 197, 192]", arg67_1: "f32[128, 197, 192]", view_170: "f16[25216, 192]", mm_79: "f16[192, 192]", view_178: "f16[25216, 576]", mm_81: "f16[576, 192]", convert_element_type_150: "f32[128, 197, 192]", arg58_1: "f32[128, 197, 192]", view_181: "f16[25216, 192]", mm_83: "f16[192, 768]", view_185: "f16[25216, 768]", mm_85: "f16[768, 192]", convert_element_type_159: "f32[128, 197, 192]", arg54_1: "f32[128, 197, 192]", view_188: "f16[25216, 192]", mm_87: "f16[192, 192]", view_196: "f16[25216, 576]", mm_89: "f16[576, 192]", convert_element_type_165: "f32[128, 197, 192]", arg45_1: "f32[128, 197, 192]", view_199: "f16[25216, 192]", mm_91: "f16[192, 768]", view_203: "f16[25216, 768]", mm_93: "f16[768, 192]", convert_element_type_174: "f32[128, 197, 192]", arg41_1: "f32[128, 197, 192]", view_206: "f16[25216, 192]", mm_95: "f16[192, 192]", view_214: "f16[25216, 576]", mm_97: "f16[576, 192]", convert_element_type_180: "f32[128, 197, 192]", mul_230: "f32[128, 197, 192]", add_48: "f32[128, 197, 192]", view_217: "f16[128, 192, 14, 14]", getitem_37: "f16[192, 3, 16, 16]"):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f16[1, 1000]" = torch.ops.aten.sum.dim_IntList(arg263_1, [0], True);  arg263_1 = None
        reshape_default: "f16[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, [1000]);  sum_dim_int_list = None
        convert_element_type_default: "f32[1000, 192]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_default_1: "f32[1000]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(select_scatter, arg188_1);  arg188_1 = None
        sum_dim_int_list_1: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[192]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None
        sum_dim_int_list_3: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True);  view_1 = None
        reshape_default_1: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [192]);  sum_dim_int_list_3 = None
        convert_element_type_default_2: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_default_3: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        sum_dim_int_list_4: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True);  view_5 = None
        reshape_default_2: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [768]);  sum_dim_int_list_4 = None
        convert_element_type_default_4: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_default_5: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        mul_tensor_1: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_9, arg184_1);  arg184_1 = None
        sum_dim_int_list_5: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_6: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [0, 1]);  convert_element_type_9 = None
        sum_dim_int_list_7: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True);  view_8 = None
        reshape_default_3: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [192]);  sum_dim_int_list_7 = None
        convert_element_type_default_6: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_default_7: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_3, torch.float32);  reshape_default_3 = None
        sum_dim_int_list_8: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_16, [0], True);  view_16 = None
        reshape_default_4: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [576]);  sum_dim_int_list_8 = None
        convert_element_type_default_8: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_default_9: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_4, torch.float32);  reshape_default_4 = None
        mul_tensor_2: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_15, arg175_1);  arg175_1 = None
        sum_dim_int_list_9: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_10: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_15, [0, 1]);  convert_element_type_15 = None
        sum_dim_int_list_11: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_19, [0], True);  view_19 = None
        reshape_default_5: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [192]);  sum_dim_int_list_11 = None
        convert_element_type_default_10: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_default_11: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_5, torch.float32);  reshape_default_5 = None
        sum_dim_int_list_12: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_23, [0], True);  view_23 = None
        reshape_default_6: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, [768]);  sum_dim_int_list_12 = None
        convert_element_type_default_12: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_default_13: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_6, torch.float32);  reshape_default_6 = None
        mul_tensor_3: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_24, arg171_1);  arg171_1 = None
        sum_dim_int_list_13: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_14: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_24, [0, 1]);  convert_element_type_24 = None
        sum_dim_int_list_15: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_26, [0], True);  view_26 = None
        reshape_default_7: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, [192]);  sum_dim_int_list_15 = None
        convert_element_type_default_14: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_default_15: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_7, torch.float32);  reshape_default_7 = None
        sum_dim_int_list_16: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_34, [0], True);  view_34 = None
        reshape_default_8: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, [576]);  sum_dim_int_list_16 = None
        convert_element_type_default_16: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_default_17: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_8, torch.float32);  reshape_default_8 = None
        mul_tensor_4: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_30, arg162_1);  arg162_1 = None
        sum_dim_int_list_17: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_18: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_30, [0, 1]);  convert_element_type_30 = None
        sum_dim_int_list_19: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True);  view_37 = None
        reshape_default_9: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, [192]);  sum_dim_int_list_19 = None
        convert_element_type_default_18: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_default_19: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_9, torch.float32);  reshape_default_9 = None
        sum_dim_int_list_20: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True);  view_41 = None
        reshape_default_10: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, [768]);  sum_dim_int_list_20 = None
        convert_element_type_default_20: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_default_21: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_10, torch.float32);  reshape_default_10 = None
        mul_tensor_5: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_39, arg158_1);  arg158_1 = None
        sum_dim_int_list_21: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_22: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_39, [0, 1]);  convert_element_type_39 = None
        sum_dim_int_list_23: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True);  view_44 = None
        reshape_default_11: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, [192]);  sum_dim_int_list_23 = None
        convert_element_type_default_22: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_default_23: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_11, torch.float32);  reshape_default_11 = None
        sum_dim_int_list_24: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_52, [0], True);  view_52 = None
        reshape_default_12: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, [576]);  sum_dim_int_list_24 = None
        convert_element_type_default_24: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_default_25: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_12, torch.float32);  reshape_default_12 = None
        mul_tensor_6: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_45, arg149_1);  arg149_1 = None
        sum_dim_int_list_25: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_26: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_45, [0, 1]);  convert_element_type_45 = None
        sum_dim_int_list_27: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_55, [0], True);  view_55 = None
        reshape_default_13: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, [192]);  sum_dim_int_list_27 = None
        convert_element_type_default_26: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_default_27: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_13, torch.float32);  reshape_default_13 = None
        sum_dim_int_list_28: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_59, [0], True);  view_59 = None
        reshape_default_14: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, [768]);  sum_dim_int_list_28 = None
        convert_element_type_default_28: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_default_29: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_14, torch.float32);  reshape_default_14 = None
        mul_tensor_7: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_54, arg145_1);  arg145_1 = None
        sum_dim_int_list_29: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_30: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_54, [0, 1]);  convert_element_type_54 = None
        sum_dim_int_list_31: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_62, [0], True);  view_62 = None
        reshape_default_15: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, [192]);  sum_dim_int_list_31 = None
        convert_element_type_default_30: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_default_31: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_15, torch.float32);  reshape_default_15 = None
        sum_dim_int_list_32: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_70, [0], True);  view_70 = None
        reshape_default_16: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, [576]);  sum_dim_int_list_32 = None
        convert_element_type_default_32: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_default_33: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_16, torch.float32);  reshape_default_16 = None
        mul_tensor_8: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_60, arg136_1);  arg136_1 = None
        sum_dim_int_list_33: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_34: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_60, [0, 1]);  convert_element_type_60 = None
        sum_dim_int_list_35: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_73, [0], True);  view_73 = None
        reshape_default_17: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, [192]);  sum_dim_int_list_35 = None
        convert_element_type_default_34: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_default_35: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_17, torch.float32);  reshape_default_17 = None
        sum_dim_int_list_36: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_77, [0], True);  view_77 = None
        reshape_default_18: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, [768]);  sum_dim_int_list_36 = None
        convert_element_type_default_36: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_default_37: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_18, torch.float32);  reshape_default_18 = None
        mul_tensor_9: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_69, arg132_1);  arg132_1 = None
        sum_dim_int_list_37: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_38: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_69, [0, 1]);  convert_element_type_69 = None
        sum_dim_int_list_39: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_80, [0], True);  view_80 = None
        reshape_default_19: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, [192]);  sum_dim_int_list_39 = None
        convert_element_type_default_38: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_default_39: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_19, torch.float32);  reshape_default_19 = None
        sum_dim_int_list_40: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_88, [0], True);  view_88 = None
        reshape_default_20: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, [576]);  sum_dim_int_list_40 = None
        convert_element_type_default_40: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_default_41: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_20, torch.float32);  reshape_default_20 = None
        mul_tensor_10: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_75, arg123_1);  arg123_1 = None
        sum_dim_int_list_41: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_42: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_75, [0, 1]);  convert_element_type_75 = None
        sum_dim_int_list_43: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_91, [0], True);  view_91 = None
        reshape_default_21: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, [192]);  sum_dim_int_list_43 = None
        convert_element_type_default_42: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_default_43: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_21, torch.float32);  reshape_default_21 = None
        sum_dim_int_list_44: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_95, [0], True);  view_95 = None
        reshape_default_22: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, [768]);  sum_dim_int_list_44 = None
        convert_element_type_default_44: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_default_45: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_22, torch.float32);  reshape_default_22 = None
        mul_tensor_11: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_84, arg119_1);  arg119_1 = None
        sum_dim_int_list_45: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_46: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_84, [0, 1]);  convert_element_type_84 = None
        sum_dim_int_list_47: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True);  view_98 = None
        reshape_default_23: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, [192]);  sum_dim_int_list_47 = None
        convert_element_type_default_46: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_default_47: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_23, torch.float32);  reshape_default_23 = None
        sum_dim_int_list_48: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_106, [0], True);  view_106 = None
        reshape_default_24: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, [576]);  sum_dim_int_list_48 = None
        convert_element_type_default_48: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_default_49: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_24, torch.float32);  reshape_default_24 = None
        mul_tensor_12: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_90, arg110_1);  arg110_1 = None
        sum_dim_int_list_49: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_50: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_90, [0, 1]);  convert_element_type_90 = None
        sum_dim_int_list_51: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_109, [0], True);  view_109 = None
        reshape_default_25: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, [192]);  sum_dim_int_list_51 = None
        convert_element_type_default_50: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_default_51: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_25, torch.float32);  reshape_default_25 = None
        sum_dim_int_list_52: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_113, [0], True);  view_113 = None
        reshape_default_26: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, [768]);  sum_dim_int_list_52 = None
        convert_element_type_default_52: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_default_53: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_26, torch.float32);  reshape_default_26 = None
        mul_tensor_13: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_99, arg106_1);  arg106_1 = None
        sum_dim_int_list_53: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_54: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_99, [0, 1]);  convert_element_type_99 = None
        sum_dim_int_list_55: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_116, [0], True);  view_116 = None
        reshape_default_27: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, [192]);  sum_dim_int_list_55 = None
        convert_element_type_default_54: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_default_55: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_27, torch.float32);  reshape_default_27 = None
        sum_dim_int_list_56: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_124, [0], True);  view_124 = None
        reshape_default_28: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, [576]);  sum_dim_int_list_56 = None
        convert_element_type_default_56: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_default_57: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_28, torch.float32);  reshape_default_28 = None
        mul_tensor_14: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_105, arg97_1);  arg97_1 = None
        sum_dim_int_list_57: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_58: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_105, [0, 1]);  convert_element_type_105 = None
        sum_dim_int_list_59: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_127, [0], True);  view_127 = None
        reshape_default_29: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, [192]);  sum_dim_int_list_59 = None
        convert_element_type_default_58: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_default_59: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_29, torch.float32);  reshape_default_29 = None
        sum_dim_int_list_60: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_131, [0], True);  view_131 = None
        reshape_default_30: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, [768]);  sum_dim_int_list_60 = None
        convert_element_type_default_60: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_default_61: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_30, torch.float32);  reshape_default_30 = None
        mul_tensor_15: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_114, arg93_1);  arg93_1 = None
        sum_dim_int_list_61: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_62: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_114, [0, 1]);  convert_element_type_114 = None
        sum_dim_int_list_63: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_134, [0], True);  view_134 = None
        reshape_default_31: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, [192]);  sum_dim_int_list_63 = None
        convert_element_type_default_62: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_default_63: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_31, torch.float32);  reshape_default_31 = None
        sum_dim_int_list_64: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_142, [0], True);  view_142 = None
        reshape_default_32: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, [576]);  sum_dim_int_list_64 = None
        convert_element_type_default_64: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_default_65: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_32, torch.float32);  reshape_default_32 = None
        mul_tensor_16: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_120, arg84_1);  arg84_1 = None
        sum_dim_int_list_65: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_66: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_120, [0, 1]);  convert_element_type_120 = None
        sum_dim_int_list_67: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_145, [0], True);  view_145 = None
        reshape_default_33: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, [192]);  sum_dim_int_list_67 = None
        convert_element_type_default_66: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_default_67: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_33, torch.float32);  reshape_default_33 = None
        sum_dim_int_list_68: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_149, [0], True);  view_149 = None
        reshape_default_34: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, [768]);  sum_dim_int_list_68 = None
        convert_element_type_default_68: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_default_69: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_34, torch.float32);  reshape_default_34 = None
        mul_tensor_17: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_129, arg80_1);  arg80_1 = None
        sum_dim_int_list_69: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_70: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_129, [0, 1]);  convert_element_type_129 = None
        sum_dim_int_list_71: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_152, [0], True);  view_152 = None
        reshape_default_35: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, [192]);  sum_dim_int_list_71 = None
        convert_element_type_default_70: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_default_71: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_35, torch.float32);  reshape_default_35 = None
        sum_dim_int_list_72: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_160, [0], True);  view_160 = None
        reshape_default_36: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, [576]);  sum_dim_int_list_72 = None
        convert_element_type_default_72: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_default_73: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_36, torch.float32);  reshape_default_36 = None
        mul_tensor_18: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_135, arg71_1);  arg71_1 = None
        sum_dim_int_list_73: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_74: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_135, [0, 1]);  convert_element_type_135 = None
        sum_dim_int_list_75: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_163, [0], True);  view_163 = None
        reshape_default_37: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, [192]);  sum_dim_int_list_75 = None
        convert_element_type_default_74: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_default_75: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_37, torch.float32);  reshape_default_37 = None
        sum_dim_int_list_76: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_167, [0], True);  view_167 = None
        reshape_default_38: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, [768]);  sum_dim_int_list_76 = None
        convert_element_type_default_76: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_default_77: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_38, torch.float32);  reshape_default_38 = None
        mul_tensor_19: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_144, arg67_1);  arg67_1 = None
        sum_dim_int_list_77: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_78: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_144, [0, 1]);  convert_element_type_144 = None
        sum_dim_int_list_79: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
        reshape_default_39: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, [192]);  sum_dim_int_list_79 = None
        convert_element_type_default_78: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_default_79: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_39, torch.float32);  reshape_default_39 = None
        sum_dim_int_list_80: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_178, [0], True);  view_178 = None
        reshape_default_40: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, [576]);  sum_dim_int_list_80 = None
        convert_element_type_default_80: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_default_81: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_40, torch.float32);  reshape_default_40 = None
        mul_tensor_20: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_150, arg58_1);  arg58_1 = None
        sum_dim_int_list_81: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_82: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_150, [0, 1]);  convert_element_type_150 = None
        sum_dim_int_list_83: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_181, [0], True);  view_181 = None
        reshape_default_41: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, [192]);  sum_dim_int_list_83 = None
        convert_element_type_default_82: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_default_83: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_41, torch.float32);  reshape_default_41 = None
        sum_dim_int_list_84: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
        reshape_default_42: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, [768]);  sum_dim_int_list_84 = None
        convert_element_type_default_84: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_default_85: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_42, torch.float32);  reshape_default_42 = None
        mul_tensor_21: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_159, arg54_1);  arg54_1 = None
        sum_dim_int_list_85: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_86: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_159, [0, 1]);  convert_element_type_159 = None
        sum_dim_int_list_87: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
        reshape_default_43: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, [192]);  sum_dim_int_list_87 = None
        convert_element_type_default_86: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_default_87: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_43, torch.float32);  reshape_default_43 = None
        sum_dim_int_list_88: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_196, [0], True);  view_196 = None
        reshape_default_44: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, [576]);  sum_dim_int_list_88 = None
        convert_element_type_default_88: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_default_89: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_44, torch.float32);  reshape_default_44 = None
        mul_tensor_22: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_165, arg45_1);  arg45_1 = None
        sum_dim_int_list_89: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_90: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_165, [0, 1]);  convert_element_type_165 = None
        sum_dim_int_list_91: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_199, [0], True);  view_199 = None
        reshape_default_45: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, [192]);  sum_dim_int_list_91 = None
        convert_element_type_default_90: "f32[192, 768]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_default_91: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_45, torch.float32);  reshape_default_45 = None
        sum_dim_int_list_92: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_203, [0], True);  view_203 = None
        reshape_default_46: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, [768]);  sum_dim_int_list_92 = None
        convert_element_type_default_92: "f32[768, 192]" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_default_93: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_46, torch.float32);  reshape_default_46 = None
        mul_tensor_23: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_174, arg41_1);  arg41_1 = None
        sum_dim_int_list_93: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_94: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_174, [0, 1]);  convert_element_type_174 = None
        sum_dim_int_list_95: "f16[1, 192]" = torch.ops.aten.sum.dim_IntList(view_206, [0], True);  view_206 = None
        reshape_default_47: "f16[192]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, [192]);  sum_dim_int_list_95 = None
        convert_element_type_default_94: "f32[192, 192]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_default_95: "f32[192]" = torch.ops.prims.convert_element_type.default(reshape_default_47, torch.float32);  reshape_default_47 = None
        sum_dim_int_list_96: "f16[1, 576]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True);  view_214 = None
        reshape_default_48: "f16[576]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, [576]);  sum_dim_int_list_96 = None
        convert_element_type_default_96: "f32[576, 192]" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_default_97: "f32[576]" = torch.ops.prims.convert_element_type.default(reshape_default_48, torch.float32);  reshape_default_48 = None
        mul_tensor_24: "f32[128, 197, 192]" = torch.ops.aten.mul.Tensor(convert_element_type_180, mul_230);  mul_230 = None
        sum_dim_int_list_97: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_98: "f32[192]" = torch.ops.aten.sum.dim_IntList(convert_element_type_180, [0, 1]);  convert_element_type_180 = None
        sum_dim_int_list_99: "f32[1, 197, 192]" = torch.ops.aten.sum.dim_IntList(add_48, [0], True)
        slice_tensor: "f32[128, 1, 192]" = torch.ops.aten.slice.Tensor(add_48, 1, 0, 1);  add_48 = None
        sum_dim_int_list_100: "f32[1, 1, 192]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0], True);  slice_tensor = None
        sum_dim_int_list_101: "f16[192]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 2, 3]);  view_217 = None
        convert_element_type_default_98: "f32[192, 3, 16, 16]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
        convert_element_type_default_99: "f32[192]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_101, torch.float32);  sum_dim_int_list_101 = None
        return (convert_element_type_default, convert_element_type_default_1, sum_dim_int_list_1, sum_dim_int_list_2, convert_element_type_default_2, convert_element_type_default_3, convert_element_type_default_4, convert_element_type_default_5, sum_dim_int_list_5, sum_dim_int_list_6, convert_element_type_default_6, convert_element_type_default_7, convert_element_type_default_8, convert_element_type_default_9, sum_dim_int_list_9, sum_dim_int_list_10, convert_element_type_default_10, convert_element_type_default_11, convert_element_type_default_12, convert_element_type_default_13, sum_dim_int_list_13, sum_dim_int_list_14, convert_element_type_default_14, convert_element_type_default_15, convert_element_type_default_16, convert_element_type_default_17, sum_dim_int_list_17, sum_dim_int_list_18, convert_element_type_default_18, convert_element_type_default_19, convert_element_type_default_20, convert_element_type_default_21, sum_dim_int_list_21, sum_dim_int_list_22, convert_element_type_default_22, convert_element_type_default_23, convert_element_type_default_24, convert_element_type_default_25, sum_dim_int_list_25, sum_dim_int_list_26, convert_element_type_default_26, convert_element_type_default_27, convert_element_type_default_28, convert_element_type_default_29, sum_dim_int_list_29, sum_dim_int_list_30, convert_element_type_default_30, convert_element_type_default_31, convert_element_type_default_32, convert_element_type_default_33, sum_dim_int_list_33, sum_dim_int_list_34, convert_element_type_default_34, convert_element_type_default_35, convert_element_type_default_36, convert_element_type_default_37, sum_dim_int_list_37, sum_dim_int_list_38, convert_element_type_default_38, convert_element_type_default_39, convert_element_type_default_40, convert_element_type_default_41, sum_dim_int_list_41, sum_dim_int_list_42, convert_element_type_default_42, convert_element_type_default_43, convert_element_type_default_44, convert_element_type_default_45, sum_dim_int_list_45, sum_dim_int_list_46, convert_element_type_default_46, convert_element_type_default_47, convert_element_type_default_48, convert_element_type_default_49, sum_dim_int_list_49, sum_dim_int_list_50, convert_element_type_default_50, convert_element_type_default_51, convert_element_type_default_52, convert_element_type_default_53, sum_dim_int_list_53, sum_dim_int_list_54, convert_element_type_default_54, convert_element_type_default_55, convert_element_type_default_56, convert_element_type_default_57, sum_dim_int_list_57, sum_dim_int_list_58, convert_element_type_default_58, convert_element_type_default_59, convert_element_type_default_60, convert_element_type_default_61, sum_dim_int_list_61, sum_dim_int_list_62, convert_element_type_default_62, convert_element_type_default_63, convert_element_type_default_64, convert_element_type_default_65, sum_dim_int_list_65, sum_dim_int_list_66, convert_element_type_default_66, convert_element_type_default_67, convert_element_type_default_68, convert_element_type_default_69, sum_dim_int_list_69, sum_dim_int_list_70, convert_element_type_default_70, convert_element_type_default_71, convert_element_type_default_72, convert_element_type_default_73, sum_dim_int_list_73, sum_dim_int_list_74, convert_element_type_default_74, convert_element_type_default_75, convert_element_type_default_76, convert_element_type_default_77, sum_dim_int_list_77, sum_dim_int_list_78, convert_element_type_default_78, convert_element_type_default_79, convert_element_type_default_80, convert_element_type_default_81, sum_dim_int_list_81, sum_dim_int_list_82, convert_element_type_default_82, convert_element_type_default_83, convert_element_type_default_84, convert_element_type_default_85, sum_dim_int_list_85, sum_dim_int_list_86, convert_element_type_default_86, convert_element_type_default_87, convert_element_type_default_88, convert_element_type_default_89, sum_dim_int_list_89, sum_dim_int_list_90, convert_element_type_default_90, convert_element_type_default_91, convert_element_type_default_92, convert_element_type_default_93, sum_dim_int_list_93, sum_dim_int_list_94, convert_element_type_default_94, convert_element_type_default_95, convert_element_type_default_96, convert_element_type_default_97, sum_dim_int_list_97, sum_dim_int_list_98, sum_dim_int_list_99, sum_dim_int_list_100, convert_element_type_default_98, convert_element_type_default_99)


def _default_make_inputs():
    return [
    torch.randn([128, 1000], dtype=torch.float16, device='cuda'),
    torch.randn([1000, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([25216, 192], dtype=torch.float16, device='cuda'),
    torch.randn([192, 192], dtype=torch.float16, device='cuda'),
    torch.randn([25216, 576], dtype=torch.float16, device='cuda'),
    torch.randn([576, 192], dtype=torch.float16, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([128, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn(4816896, dtype=torch.float16, device='cuda').as_strided([128, 192, 14, 14], [37632, 1, 2688, 192]),  # view_217
    torch.randn([192, 3, 16, 16], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
