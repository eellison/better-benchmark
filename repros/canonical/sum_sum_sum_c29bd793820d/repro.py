"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g53
Pattern hash: c29bd793820d
Shape hash: adab2c3d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, div: "f16[8, 1000]", mm_1: "f16[1000, 768]", mm_3: "f16[1000, 768]", add: "f32[8, 198, 768]", arg186_1: "f32[8, 198, 768]", view_1: "f16[1584, 768]", mm_5: "f16[768, 3072]", view_5: "f16[1584, 3072]", mm_7: "f16[3072, 768]", convert_element_type_12: "f32[8, 198, 768]", arg182_1: "f32[8, 198, 768]", view_8: "f16[1584, 768]", mm_9: "f16[768, 768]", view_16: "f16[1584, 2304]", mm_11: "f16[2304, 768]", convert_element_type_18: "f32[8, 198, 768]", arg173_1: "f32[8, 198, 768]", view_19: "f16[1584, 768]", mm_13: "f16[768, 3072]", view_23: "f16[1584, 3072]", mm_15: "f16[3072, 768]", convert_element_type_27: "f32[8, 198, 768]", arg169_1: "f32[8, 198, 768]", view_26: "f16[1584, 768]", mm_17: "f16[768, 768]", view_34: "f16[1584, 2304]", mm_19: "f16[2304, 768]", convert_element_type_33: "f32[8, 198, 768]", arg160_1: "f32[8, 198, 768]", view_37: "f16[1584, 768]", mm_21: "f16[768, 3072]", view_41: "f16[1584, 3072]", mm_23: "f16[3072, 768]", convert_element_type_42: "f32[8, 198, 768]", arg156_1: "f32[8, 198, 768]", view_44: "f16[1584, 768]", mm_25: "f16[768, 768]", view_52: "f16[1584, 2304]", mm_27: "f16[2304, 768]", convert_element_type_48: "f32[8, 198, 768]", arg147_1: "f32[8, 198, 768]", view_55: "f16[1584, 768]", mm_29: "f16[768, 3072]", view_59: "f16[1584, 3072]", mm_31: "f16[3072, 768]", convert_element_type_57: "f32[8, 198, 768]", arg143_1: "f32[8, 198, 768]", view_62: "f16[1584, 768]", mm_33: "f16[768, 768]", view_70: "f16[1584, 2304]", mm_35: "f16[2304, 768]", convert_element_type_63: "f32[8, 198, 768]", arg134_1: "f32[8, 198, 768]", view_73: "f16[1584, 768]", mm_37: "f16[768, 3072]", view_77: "f16[1584, 3072]", mm_39: "f16[3072, 768]", convert_element_type_72: "f32[8, 198, 768]", arg130_1: "f32[8, 198, 768]", view_80: "f16[1584, 768]", mm_41: "f16[768, 768]", view_88: "f16[1584, 2304]", mm_43: "f16[2304, 768]", convert_element_type_78: "f32[8, 198, 768]", arg121_1: "f32[8, 198, 768]", view_91: "f16[1584, 768]", mm_45: "f16[768, 3072]", view_95: "f16[1584, 3072]", mm_47: "f16[3072, 768]", convert_element_type_87: "f32[8, 198, 768]", arg117_1: "f32[8, 198, 768]", view_98: "f16[1584, 768]", mm_49: "f16[768, 768]", view_106: "f16[1584, 2304]", mm_51: "f16[2304, 768]", convert_element_type_93: "f32[8, 198, 768]", arg108_1: "f32[8, 198, 768]", view_109: "f16[1584, 768]", mm_53: "f16[768, 3072]", view_113: "f16[1584, 3072]", mm_55: "f16[3072, 768]", convert_element_type_102: "f32[8, 198, 768]", arg104_1: "f32[8, 198, 768]", view_116: "f16[1584, 768]", mm_57: "f16[768, 768]", view_124: "f16[1584, 2304]", mm_59: "f16[2304, 768]", convert_element_type_108: "f32[8, 198, 768]", arg95_1: "f32[8, 198, 768]", view_127: "f16[1584, 768]", mm_61: "f16[768, 3072]", view_131: "f16[1584, 3072]", mm_63: "f16[3072, 768]", convert_element_type_117: "f32[8, 198, 768]", arg91_1: "f32[8, 198, 768]", view_134: "f16[1584, 768]", mm_65: "f16[768, 768]", view_142: "f16[1584, 2304]", mm_67: "f16[2304, 768]", convert_element_type_123: "f32[8, 198, 768]", arg82_1: "f32[8, 198, 768]", view_145: "f16[1584, 768]", mm_69: "f16[768, 3072]", view_149: "f16[1584, 3072]", mm_71: "f16[3072, 768]", convert_element_type_132: "f32[8, 198, 768]", arg78_1: "f32[8, 198, 768]", view_152: "f16[1584, 768]", mm_73: "f16[768, 768]", view_160: "f16[1584, 2304]", mm_75: "f16[2304, 768]", convert_element_type_138: "f32[8, 198, 768]", arg69_1: "f32[8, 198, 768]", view_163: "f16[1584, 768]", mm_77: "f16[768, 3072]", view_167: "f16[1584, 3072]", mm_79: "f16[3072, 768]", convert_element_type_147: "f32[8, 198, 768]", arg65_1: "f32[8, 198, 768]", view_170: "f16[1584, 768]", mm_81: "f16[768, 768]", view_178: "f16[1584, 2304]", mm_83: "f16[2304, 768]", convert_element_type_153: "f32[8, 198, 768]", arg56_1: "f32[8, 198, 768]", view_181: "f16[1584, 768]", mm_85: "f16[768, 3072]", view_185: "f16[1584, 3072]", mm_87: "f16[3072, 768]", convert_element_type_162: "f32[8, 198, 768]", arg52_1: "f32[8, 198, 768]", view_188: "f16[1584, 768]", mm_89: "f16[768, 768]", view_196: "f16[1584, 2304]", mm_91: "f16[2304, 768]", convert_element_type_168: "f32[8, 198, 768]", arg43_1: "f32[8, 198, 768]", view_199: "f16[1584, 768]", mm_93: "f16[768, 3072]", view_203: "f16[1584, 3072]", mm_95: "f16[3072, 768]", convert_element_type_177: "f32[8, 198, 768]", arg39_1: "f32[8, 198, 768]", view_206: "f16[1584, 768]", mm_97: "f16[768, 768]", view_214: "f16[1584, 2304]", mm_99: "f16[2304, 768]", convert_element_type_183: "f32[8, 198, 768]", mul_230: "f32[8, 198, 768]", add_49: "f32[8, 198, 768]", view_217: "f16[8, 768, 14, 14]", getitem_37: "f16[768, 3, 16, 16]"):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f16[1, 1000]" = torch.ops.aten.sum.dim_IntList(div, [0], True);  div = None
        reshape_default: "f16[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, [1000]);  sum_dim_int_list = None
        convert_element_type_default: "f32[1000, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_default_1: "f32[1000]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32)
        convert_element_type_default_2: "f32[1000, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_default_3: "f32[1000]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(add, arg186_1);  arg186_1 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(add, [0, 1]);  add = None
        sum_dim_int_list_3: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True);  view_1 = None
        reshape_default_1: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [768]);  sum_dim_int_list_3 = None
        convert_element_type_default_4: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_default_5: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        sum_dim_int_list_4: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True);  view_5 = None
        reshape_default_2: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [3072]);  sum_dim_int_list_4 = None
        convert_element_type_default_6: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_default_7: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        mul_tensor_1: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, arg182_1);  arg182_1 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_12, [0, 1]);  convert_element_type_12 = None
        sum_dim_int_list_7: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True);  view_8 = None
        reshape_default_3: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [768]);  sum_dim_int_list_7 = None
        convert_element_type_default_8: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_default_9: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_3, torch.float32);  reshape_default_3 = None
        sum_dim_int_list_8: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_16, [0], True);  view_16 = None
        reshape_default_4: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [2304]);  sum_dim_int_list_8 = None
        convert_element_type_default_10: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_default_11: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_4, torch.float32);  reshape_default_4 = None
        mul_tensor_2: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_18, arg173_1);  arg173_1 = None
        sum_dim_int_list_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_18, [0, 1]);  convert_element_type_18 = None
        sum_dim_int_list_11: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_19, [0], True);  view_19 = None
        reshape_default_5: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [768]);  sum_dim_int_list_11 = None
        convert_element_type_default_12: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_default_13: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_5, torch.float32);  reshape_default_5 = None
        sum_dim_int_list_12: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_23, [0], True);  view_23 = None
        reshape_default_6: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, [3072]);  sum_dim_int_list_12 = None
        convert_element_type_default_14: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_default_15: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_6, torch.float32);  reshape_default_6 = None
        mul_tensor_3: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_27, arg169_1);  arg169_1 = None
        sum_dim_int_list_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_27, [0, 1]);  convert_element_type_27 = None
        sum_dim_int_list_15: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_26, [0], True);  view_26 = None
        reshape_default_7: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, [768]);  sum_dim_int_list_15 = None
        convert_element_type_default_16: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_default_17: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_7, torch.float32);  reshape_default_7 = None
        sum_dim_int_list_16: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_34, [0], True);  view_34 = None
        reshape_default_8: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, [2304]);  sum_dim_int_list_16 = None
        convert_element_type_default_18: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_default_19: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_8, torch.float32);  reshape_default_8 = None
        mul_tensor_4: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_33, arg160_1);  arg160_1 = None
        sum_dim_int_list_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_33, [0, 1]);  convert_element_type_33 = None
        sum_dim_int_list_19: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True);  view_37 = None
        reshape_default_9: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, [768]);  sum_dim_int_list_19 = None
        convert_element_type_default_20: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_default_21: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_9, torch.float32);  reshape_default_9 = None
        sum_dim_int_list_20: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True);  view_41 = None
        reshape_default_10: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, [3072]);  sum_dim_int_list_20 = None
        convert_element_type_default_22: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_default_23: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_10, torch.float32);  reshape_default_10 = None
        mul_tensor_5: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_42, arg156_1);  arg156_1 = None
        sum_dim_int_list_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_42, [0, 1]);  convert_element_type_42 = None
        sum_dim_int_list_23: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_44, [0], True);  view_44 = None
        reshape_default_11: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, [768]);  sum_dim_int_list_23 = None
        convert_element_type_default_24: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_default_25: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_11, torch.float32);  reshape_default_11 = None
        sum_dim_int_list_24: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_52, [0], True);  view_52 = None
        reshape_default_12: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, [2304]);  sum_dim_int_list_24 = None
        convert_element_type_default_26: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_default_27: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_12, torch.float32);  reshape_default_12 = None
        mul_tensor_6: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_48, arg147_1);  arg147_1 = None
        sum_dim_int_list_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_48, [0, 1]);  convert_element_type_48 = None
        sum_dim_int_list_27: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_55, [0], True);  view_55 = None
        reshape_default_13: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, [768]);  sum_dim_int_list_27 = None
        convert_element_type_default_28: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_default_29: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_13, torch.float32);  reshape_default_13 = None
        sum_dim_int_list_28: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_59, [0], True);  view_59 = None
        reshape_default_14: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, [3072]);  sum_dim_int_list_28 = None
        convert_element_type_default_30: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_default_31: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_14, torch.float32);  reshape_default_14 = None
        mul_tensor_7: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_57, arg143_1);  arg143_1 = None
        sum_dim_int_list_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_57, [0, 1]);  convert_element_type_57 = None
        sum_dim_int_list_31: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_62, [0], True);  view_62 = None
        reshape_default_15: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, [768]);  sum_dim_int_list_31 = None
        convert_element_type_default_32: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_default_33: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_15, torch.float32);  reshape_default_15 = None
        sum_dim_int_list_32: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_70, [0], True);  view_70 = None
        reshape_default_16: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, [2304]);  sum_dim_int_list_32 = None
        convert_element_type_default_34: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_default_35: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_16, torch.float32);  reshape_default_16 = None
        mul_tensor_8: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_63, arg134_1);  arg134_1 = None
        sum_dim_int_list_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_63, [0, 1]);  convert_element_type_63 = None
        sum_dim_int_list_35: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_73, [0], True);  view_73 = None
        reshape_default_17: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, [768]);  sum_dim_int_list_35 = None
        convert_element_type_default_36: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_default_37: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_17, torch.float32);  reshape_default_17 = None
        sum_dim_int_list_36: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_77, [0], True);  view_77 = None
        reshape_default_18: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, [3072]);  sum_dim_int_list_36 = None
        convert_element_type_default_38: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_default_39: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_18, torch.float32);  reshape_default_18 = None
        mul_tensor_9: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_72, arg130_1);  arg130_1 = None
        sum_dim_int_list_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_72, [0, 1]);  convert_element_type_72 = None
        sum_dim_int_list_39: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_80, [0], True);  view_80 = None
        reshape_default_19: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, [768]);  sum_dim_int_list_39 = None
        convert_element_type_default_40: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_default_41: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_19, torch.float32);  reshape_default_19 = None
        sum_dim_int_list_40: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_88, [0], True);  view_88 = None
        reshape_default_20: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, [2304]);  sum_dim_int_list_40 = None
        convert_element_type_default_42: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_default_43: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_20, torch.float32);  reshape_default_20 = None
        mul_tensor_10: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_78, arg121_1);  arg121_1 = None
        sum_dim_int_list_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_78, [0, 1]);  convert_element_type_78 = None
        sum_dim_int_list_43: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_91, [0], True);  view_91 = None
        reshape_default_21: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, [768]);  sum_dim_int_list_43 = None
        convert_element_type_default_44: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_default_45: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_21, torch.float32);  reshape_default_21 = None
        sum_dim_int_list_44: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_95, [0], True);  view_95 = None
        reshape_default_22: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, [3072]);  sum_dim_int_list_44 = None
        convert_element_type_default_46: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_default_47: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_22, torch.float32);  reshape_default_22 = None
        mul_tensor_11: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_87, arg117_1);  arg117_1 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_87, [0, 1]);  convert_element_type_87 = None
        sum_dim_int_list_47: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True);  view_98 = None
        reshape_default_23: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, [768]);  sum_dim_int_list_47 = None
        convert_element_type_default_48: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_default_49: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_23, torch.float32);  reshape_default_23 = None
        sum_dim_int_list_48: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_106, [0], True);  view_106 = None
        reshape_default_24: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, [2304]);  sum_dim_int_list_48 = None
        convert_element_type_default_50: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_default_51: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_24, torch.float32);  reshape_default_24 = None
        mul_tensor_12: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_93, arg108_1);  arg108_1 = None
        sum_dim_int_list_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_93, [0, 1]);  convert_element_type_93 = None
        sum_dim_int_list_51: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_109, [0], True);  view_109 = None
        reshape_default_25: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, [768]);  sum_dim_int_list_51 = None
        convert_element_type_default_52: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_default_53: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_25, torch.float32);  reshape_default_25 = None
        sum_dim_int_list_52: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_113, [0], True);  view_113 = None
        reshape_default_26: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, [3072]);  sum_dim_int_list_52 = None
        convert_element_type_default_54: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_default_55: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_26, torch.float32);  reshape_default_26 = None
        mul_tensor_13: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_102, arg104_1);  arg104_1 = None
        sum_dim_int_list_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_102, [0, 1]);  convert_element_type_102 = None
        sum_dim_int_list_55: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_116, [0], True);  view_116 = None
        reshape_default_27: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, [768]);  sum_dim_int_list_55 = None
        convert_element_type_default_56: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_default_57: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_27, torch.float32);  reshape_default_27 = None
        sum_dim_int_list_56: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_124, [0], True);  view_124 = None
        reshape_default_28: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, [2304]);  sum_dim_int_list_56 = None
        convert_element_type_default_58: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_default_59: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_28, torch.float32);  reshape_default_28 = None
        mul_tensor_14: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_108, arg95_1);  arg95_1 = None
        sum_dim_int_list_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_108, [0, 1]);  convert_element_type_108 = None
        sum_dim_int_list_59: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_127, [0], True);  view_127 = None
        reshape_default_29: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, [768]);  sum_dim_int_list_59 = None
        convert_element_type_default_60: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_default_61: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_29, torch.float32);  reshape_default_29 = None
        sum_dim_int_list_60: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_131, [0], True);  view_131 = None
        reshape_default_30: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, [3072]);  sum_dim_int_list_60 = None
        convert_element_type_default_62: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_default_63: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_30, torch.float32);  reshape_default_30 = None
        mul_tensor_15: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_117, arg91_1);  arg91_1 = None
        sum_dim_int_list_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_117, [0, 1]);  convert_element_type_117 = None
        sum_dim_int_list_63: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_134, [0], True);  view_134 = None
        reshape_default_31: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, [768]);  sum_dim_int_list_63 = None
        convert_element_type_default_64: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_default_65: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_31, torch.float32);  reshape_default_31 = None
        sum_dim_int_list_64: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_142, [0], True);  view_142 = None
        reshape_default_32: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, [2304]);  sum_dim_int_list_64 = None
        convert_element_type_default_66: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_default_67: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_32, torch.float32);  reshape_default_32 = None
        mul_tensor_16: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_123, arg82_1);  arg82_1 = None
        sum_dim_int_list_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_123, [0, 1]);  convert_element_type_123 = None
        sum_dim_int_list_67: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_145, [0], True);  view_145 = None
        reshape_default_33: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, [768]);  sum_dim_int_list_67 = None
        convert_element_type_default_68: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_default_69: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_33, torch.float32);  reshape_default_33 = None
        sum_dim_int_list_68: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_149, [0], True);  view_149 = None
        reshape_default_34: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, [3072]);  sum_dim_int_list_68 = None
        convert_element_type_default_70: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_default_71: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_34, torch.float32);  reshape_default_34 = None
        mul_tensor_17: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_132, arg78_1);  arg78_1 = None
        sum_dim_int_list_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_132, [0, 1]);  convert_element_type_132 = None
        sum_dim_int_list_71: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_152, [0], True);  view_152 = None
        reshape_default_35: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, [768]);  sum_dim_int_list_71 = None
        convert_element_type_default_72: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_default_73: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_35, torch.float32);  reshape_default_35 = None
        sum_dim_int_list_72: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_160, [0], True);  view_160 = None
        reshape_default_36: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, [2304]);  sum_dim_int_list_72 = None
        convert_element_type_default_74: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_default_75: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_36, torch.float32);  reshape_default_36 = None
        mul_tensor_18: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_138, arg69_1);  arg69_1 = None
        sum_dim_int_list_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_138, [0, 1]);  convert_element_type_138 = None
        sum_dim_int_list_75: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_163, [0], True);  view_163 = None
        reshape_default_37: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, [768]);  sum_dim_int_list_75 = None
        convert_element_type_default_76: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_default_77: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_37, torch.float32);  reshape_default_37 = None
        sum_dim_int_list_76: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_167, [0], True);  view_167 = None
        reshape_default_38: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, [3072]);  sum_dim_int_list_76 = None
        convert_element_type_default_78: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_default_79: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_38, torch.float32);  reshape_default_38 = None
        mul_tensor_19: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_147, arg65_1);  arg65_1 = None
        sum_dim_int_list_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_147, [0, 1]);  convert_element_type_147 = None
        sum_dim_int_list_79: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
        reshape_default_39: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, [768]);  sum_dim_int_list_79 = None
        convert_element_type_default_80: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_default_81: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_39, torch.float32);  reshape_default_39 = None
        sum_dim_int_list_80: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_178, [0], True);  view_178 = None
        reshape_default_40: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, [2304]);  sum_dim_int_list_80 = None
        convert_element_type_default_82: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_default_83: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_40, torch.float32);  reshape_default_40 = None
        mul_tensor_20: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_153, arg56_1);  arg56_1 = None
        sum_dim_int_list_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_153, [0, 1]);  convert_element_type_153 = None
        sum_dim_int_list_83: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_181, [0], True);  view_181 = None
        reshape_default_41: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, [768]);  sum_dim_int_list_83 = None
        convert_element_type_default_84: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_default_85: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_41, torch.float32);  reshape_default_41 = None
        sum_dim_int_list_84: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
        reshape_default_42: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, [3072]);  sum_dim_int_list_84 = None
        convert_element_type_default_86: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_default_87: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_42, torch.float32);  reshape_default_42 = None
        mul_tensor_21: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_162, arg52_1);  arg52_1 = None
        sum_dim_int_list_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_162, [0, 1]);  convert_element_type_162 = None
        sum_dim_int_list_87: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
        reshape_default_43: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, [768]);  sum_dim_int_list_87 = None
        convert_element_type_default_88: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_default_89: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_43, torch.float32);  reshape_default_43 = None
        sum_dim_int_list_88: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_196, [0], True);  view_196 = None
        reshape_default_44: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, [2304]);  sum_dim_int_list_88 = None
        convert_element_type_default_90: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_default_91: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_44, torch.float32);  reshape_default_44 = None
        mul_tensor_22: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_168, arg43_1);  arg43_1 = None
        sum_dim_int_list_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_168, [0, 1]);  convert_element_type_168 = None
        sum_dim_int_list_91: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_199, [0], True);  view_199 = None
        reshape_default_45: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, [768]);  sum_dim_int_list_91 = None
        convert_element_type_default_92: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_default_93: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_45, torch.float32);  reshape_default_45 = None
        sum_dim_int_list_92: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_203, [0], True);  view_203 = None
        reshape_default_46: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, [3072]);  sum_dim_int_list_92 = None
        convert_element_type_default_94: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_default_95: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_46, torch.float32);  reshape_default_46 = None
        mul_tensor_23: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_177, arg39_1);  arg39_1 = None
        sum_dim_int_list_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_177, [0, 1]);  convert_element_type_177 = None
        sum_dim_int_list_95: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_206, [0], True);  view_206 = None
        reshape_default_47: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, [768]);  sum_dim_int_list_95 = None
        convert_element_type_default_96: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_default_97: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_47, torch.float32);  reshape_default_47 = None
        sum_dim_int_list_96: "f16[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True);  view_214 = None
        reshape_default_48: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, [2304]);  sum_dim_int_list_96 = None
        convert_element_type_default_98: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_default_99: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_48, torch.float32);  reshape_default_48 = None
        mul_tensor_24: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_183, mul_230);  mul_230 = None
        sum_dim_int_list_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_183, [0, 1]);  convert_element_type_183 = None
        sum_dim_int_list_99: "f32[1, 198, 768]" = torch.ops.aten.sum.dim_IntList(add_49, [0], True)
        slice_tensor: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_49, 1, 0, 1)
        slice_tensor_1: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_49, 1, 1, 2);  add_49 = None
        sum_dim_int_list_100: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_tensor_1, [0], True);  slice_tensor_1 = None
        sum_dim_int_list_101: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0], True);  slice_tensor = None
        sum_dim_int_list_102: "f16[768]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 2, 3]);  view_217 = None
        convert_element_type_default_100: "f32[768, 3, 16, 16]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
        convert_element_type_default_101: "f32[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_102, torch.float32);  sum_dim_int_list_102 = None
        return (convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, convert_element_type_default_3, sum_dim_int_list_1, sum_dim_int_list_2, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_7, sum_dim_int_list_5, sum_dim_int_list_6, convert_element_type_default_8, convert_element_type_default_9, convert_element_type_default_10, convert_element_type_default_11, sum_dim_int_list_9, sum_dim_int_list_10, convert_element_type_default_12, convert_element_type_default_13, convert_element_type_default_14, convert_element_type_default_15, sum_dim_int_list_13, sum_dim_int_list_14, convert_element_type_default_16, convert_element_type_default_17, convert_element_type_default_18, convert_element_type_default_19, sum_dim_int_list_17, sum_dim_int_list_18, convert_element_type_default_20, convert_element_type_default_21, convert_element_type_default_22, convert_element_type_default_23, sum_dim_int_list_21, sum_dim_int_list_22, convert_element_type_default_24, convert_element_type_default_25, convert_element_type_default_26, convert_element_type_default_27, sum_dim_int_list_25, sum_dim_int_list_26, convert_element_type_default_28, convert_element_type_default_29, convert_element_type_default_30, convert_element_type_default_31, sum_dim_int_list_29, sum_dim_int_list_30, convert_element_type_default_32, convert_element_type_default_33, convert_element_type_default_34, convert_element_type_default_35, sum_dim_int_list_33, sum_dim_int_list_34, convert_element_type_default_36, convert_element_type_default_37, convert_element_type_default_38, convert_element_type_default_39, sum_dim_int_list_37, sum_dim_int_list_38, convert_element_type_default_40, convert_element_type_default_41, convert_element_type_default_42, convert_element_type_default_43, sum_dim_int_list_41, sum_dim_int_list_42, convert_element_type_default_44, convert_element_type_default_45, convert_element_type_default_46, convert_element_type_default_47, sum_dim_int_list_45, sum_dim_int_list_46, convert_element_type_default_48, convert_element_type_default_49, convert_element_type_default_50, convert_element_type_default_51, sum_dim_int_list_49, sum_dim_int_list_50, convert_element_type_default_52, convert_element_type_default_53, convert_element_type_default_54, convert_element_type_default_55, sum_dim_int_list_53, sum_dim_int_list_54, convert_element_type_default_56, convert_element_type_default_57, convert_element_type_default_58, convert_element_type_default_59, sum_dim_int_list_57, sum_dim_int_list_58, convert_element_type_default_60, convert_element_type_default_61, convert_element_type_default_62, convert_element_type_default_63, sum_dim_int_list_61, sum_dim_int_list_62, convert_element_type_default_64, convert_element_type_default_65, convert_element_type_default_66, convert_element_type_default_67, sum_dim_int_list_65, sum_dim_int_list_66, convert_element_type_default_68, convert_element_type_default_69, convert_element_type_default_70, convert_element_type_default_71, sum_dim_int_list_69, sum_dim_int_list_70, convert_element_type_default_72, convert_element_type_default_73, convert_element_type_default_74, convert_element_type_default_75, sum_dim_int_list_73, sum_dim_int_list_74, convert_element_type_default_76, convert_element_type_default_77, convert_element_type_default_78, convert_element_type_default_79, sum_dim_int_list_77, sum_dim_int_list_78, convert_element_type_default_80, convert_element_type_default_81, convert_element_type_default_82, convert_element_type_default_83, sum_dim_int_list_81, sum_dim_int_list_82, convert_element_type_default_84, convert_element_type_default_85, convert_element_type_default_86, convert_element_type_default_87, sum_dim_int_list_85, sum_dim_int_list_86, convert_element_type_default_88, convert_element_type_default_89, convert_element_type_default_90, convert_element_type_default_91, sum_dim_int_list_89, sum_dim_int_list_90, convert_element_type_default_92, convert_element_type_default_93, convert_element_type_default_94, convert_element_type_default_95, sum_dim_int_list_93, sum_dim_int_list_94, convert_element_type_default_96, convert_element_type_default_97, convert_element_type_default_98, convert_element_type_default_99, sum_dim_int_list_97, sum_dim_int_list_98, sum_dim_int_list_99, sum_dim_int_list_100, sum_dim_int_list_101, convert_element_type_default_100, convert_element_type_default_101)


def _default_make_inputs():
    return [
    torch.randn([8, 1000], dtype=torch.float16, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn(1204224, dtype=torch.float16, device='cuda').as_strided([8, 768, 14, 14], [150528, 1, 10752, 768]),  # view_217
    torch.randn([768, 3, 16, 16], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
