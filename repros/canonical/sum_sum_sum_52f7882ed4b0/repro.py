"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s3_g174
Pattern hash: 52f7882ed4b0
Shape hash: c1d0c88d
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
    def forward(self, convert_element_type_2: "f16[4, 768]", mm_1: "f16[768, 768]", add: "f32[4, 476, 768]", arg127_1: "f32[4, 476, 768]", view_1: "f16[1904, 768]", mm_3: "f16[768, 3072]", view_5: "f16[1904, 3072]", mm_5: "f16[3072, 768]", add_3: "f32[4, 476, 768]", arg123_1: "f32[4, 476, 768]", view_8: "f16[1904, 768]", mm_7: "f16[768, 768]", view_21: "f16[1904, 768]", mm_9: "f16[768, 768]", view_24: "f16[1904, 768]", mm_11: "f16[768, 768]", view_27: "f16[1904, 768]", mm_13: "f16[768, 768]", add_7: "f32[4, 476, 768]", arg119_1: "f32[4, 476, 768]", view_30: "f16[1904, 768]", mm_15: "f16[768, 3072]", view_34: "f16[1904, 3072]", mm_17: "f16[3072, 768]", add_10: "f32[4, 476, 768]", arg115_1: "f32[4, 476, 768]", view_37: "f16[1904, 768]", mm_19: "f16[768, 768]", view_50: "f16[1904, 768]", mm_21: "f16[768, 768]", view_53: "f16[1904, 768]", mm_23: "f16[768, 768]", view_56: "f16[1904, 768]", mm_25: "f16[768, 768]", add_14: "f32[4, 476, 768]", arg111_1: "f32[4, 476, 768]", view_59: "f16[1904, 768]", mm_27: "f16[768, 3072]", view_63: "f16[1904, 3072]", mm_29: "f16[3072, 768]", add_17: "f32[4, 476, 768]", arg107_1: "f32[4, 476, 768]", view_66: "f16[1904, 768]", mm_31: "f16[768, 768]", view_79: "f16[1904, 768]", mm_33: "f16[768, 768]", view_82: "f16[1904, 768]", mm_35: "f16[768, 768]", view_85: "f16[1904, 768]", mm_37: "f16[768, 768]", add_21: "f32[4, 476, 768]", arg103_1: "f32[4, 476, 768]", view_88: "f16[1904, 768]", mm_39: "f16[768, 3072]", view_92: "f16[1904, 3072]", mm_41: "f16[3072, 768]", add_24: "f32[4, 476, 768]", arg99_1: "f32[4, 476, 768]", view_95: "f16[1904, 768]", mm_43: "f16[768, 768]", view_108: "f16[1904, 768]", mm_45: "f16[768, 768]", view_111: "f16[1904, 768]", mm_47: "f16[768, 768]", view_114: "f16[1904, 768]", mm_49: "f16[768, 768]", add_28: "f32[4, 476, 768]", arg95_1: "f32[4, 476, 768]", view_117: "f16[1904, 768]", mm_51: "f16[768, 3072]", view_121: "f16[1904, 3072]", mm_53: "f16[3072, 768]", add_31: "f32[4, 476, 768]", arg91_1: "f32[4, 476, 768]", view_124: "f16[1904, 768]", mm_55: "f16[768, 768]", view_137: "f16[1904, 768]", mm_57: "f16[768, 768]", view_140: "f16[1904, 768]", mm_59: "f16[768, 768]", view_143: "f16[1904, 768]", mm_61: "f16[768, 768]", add_35: "f32[4, 476, 768]", arg87_1: "f32[4, 476, 768]", view_146: "f16[1904, 768]", mm_63: "f16[768, 3072]", view_150: "f16[1904, 3072]", mm_65: "f16[3072, 768]", add_38: "f32[4, 476, 768]", arg83_1: "f32[4, 476, 768]", view_153: "f16[1904, 768]", mm_67: "f16[768, 768]", view_166: "f16[1904, 768]", mm_69: "f16[768, 768]", view_169: "f16[1904, 768]", mm_71: "f16[768, 768]", view_172: "f16[1904, 768]", mm_73: "f16[768, 768]", add_42: "f32[4, 476, 768]", arg79_1: "f32[4, 476, 768]", view_175: "f16[1904, 768]", mm_75: "f16[768, 3072]", view_179: "f16[1904, 3072]", mm_77: "f16[3072, 768]", add_45: "f32[4, 476, 768]", arg75_1: "f32[4, 476, 768]", view_182: "f16[1904, 768]", mm_79: "f16[768, 768]", view_195: "f16[1904, 768]", mm_81: "f16[768, 768]", view_198: "f16[1904, 768]", mm_83: "f16[768, 768]", view_201: "f16[1904, 768]", mm_85: "f16[768, 768]", add_49: "f32[4, 476, 768]", arg71_1: "f32[4, 476, 768]", view_204: "f16[1904, 768]", mm_87: "f16[768, 3072]", view_208: "f16[1904, 3072]", mm_89: "f16[3072, 768]", add_52: "f32[4, 476, 768]", arg67_1: "f32[4, 476, 768]", view_211: "f16[1904, 768]", mm_91: "f16[768, 768]", view_224: "f16[1904, 768]", mm_93: "f16[768, 768]", view_227: "f16[1904, 768]", mm_95: "f16[768, 768]", view_230: "f16[1904, 768]", mm_97: "f16[768, 768]", add_56: "f32[4, 476, 768]", arg63_1: "f32[4, 476, 768]", view_233: "f16[1904, 768]", mm_99: "f16[768, 3072]", view_237: "f16[1904, 3072]", mm_101: "f16[3072, 768]", add_59: "f32[4, 476, 768]", arg59_1: "f32[4, 476, 768]", view_240: "f16[1904, 768]", mm_103: "f16[768, 768]", view_253: "f16[1904, 768]", mm_105: "f16[768, 768]", view_256: "f16[1904, 768]", mm_107: "f16[768, 768]", view_259: "f16[1904, 768]", mm_109: "f16[768, 768]", add_63: "f32[4, 476, 768]", arg55_1: "f32[4, 476, 768]", view_262: "f16[1904, 768]", mm_111: "f16[768, 3072]", view_266: "f16[1904, 3072]", mm_113: "f16[3072, 768]", add_66: "f32[4, 476, 768]", arg51_1: "f32[4, 476, 768]", view_269: "f16[1904, 768]", mm_115: "f16[768, 768]", view_282: "f16[1904, 768]", mm_117: "f16[768, 768]", view_285: "f16[1904, 768]", mm_119: "f16[768, 768]", view_288: "f16[1904, 768]", mm_121: "f16[768, 768]", add_70: "f32[4, 476, 768]", arg47_1: "f32[4, 476, 768]", view_291: "f16[1904, 768]", mm_123: "f16[768, 3072]", view_295: "f16[1904, 3072]", mm_125: "f16[3072, 768]", add_73: "f32[4, 476, 768]", arg43_1: "f32[4, 476, 768]", view_298: "f16[1904, 768]", mm_127: "f16[768, 768]", view_311: "f16[1904, 768]", mm_129: "f16[768, 768]", view_314: "f16[1904, 768]", mm_131: "f16[768, 768]", view_317: "f16[1904, 768]", mm_133: "f16[768, 768]", add_77: "f32[4, 476, 768]", arg39_1: "f32[4, 476, 768]", view_320: "f16[1904, 768]", mm_135: "f16[768, 3072]", view_324: "f16[1904, 3072]", mm_137: "f16[3072, 768]", add_80: "f32[4, 476, 768]", arg35_1: "f32[4, 476, 768]", arg276_1: "f32[4, 476, 768]", mul_227: "f32[4, 476, 768]", view_327: "f16[1904, 768]", mm_139: "f16[768, 768]", view_341: "f16[1904, 768]", mm_140: "f16[1904, 768]", mm_141: "f16[768, 768]", view_344: "f16[1904, 768]", mm_142: "f16[1904, 768]", mm_143: "f16[768, 768]", view_347: "f16[1904, 768]", mm_144: "f16[1904, 768]", mm_145: "f16[768, 768]", arg2_1: "f32[768]", arg29_1: "f32[4, 476, 768]", arg275_1: "f32[4, 476, 1]", arg0_1: "i64[4, 476]", arg28_1: "i64[1, 476]", arg1_1: "i64[4, 476]"):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0], True, dtype = torch.float32);  convert_element_type_2 = None
        reshape_default: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list, [768]);  sum_dim_int_list = None
        convert_element_type_default: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_default_1: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add, arg127_1);  arg127_1 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(add, [0, 1]);  add = None
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32);  view_1 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [768]);  sum_dim_int_list_3 = None
        convert_element_type_default_2: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_default_3: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        sum_dim_int_list_4: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True, dtype = torch.float32);  view_5 = None
        reshape_default_2: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [3072]);  sum_dim_int_list_4 = None
        convert_element_type_default_4: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_default_5: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        mul_tensor_1: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_3, arg123_1);  arg123_1 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_3, [0, 1]);  add_3 = None
        sum_dim_int_list_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        reshape_default_3: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [768]);  sum_dim_int_list_7 = None
        convert_element_type_default_6: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_default_7: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_3, torch.float32);  reshape_default_3 = None
        sum_dim_int_list_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_21, [0], True, dtype = torch.float32);  view_21 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [768]);  sum_dim_int_list_8 = None
        convert_element_type_default_8: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_default_9: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_4, torch.float32);  reshape_default_4 = None
        sum_dim_int_list_9: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_24, [0], True, dtype = torch.float32);  view_24 = None
        reshape_default_5: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, [768]);  sum_dim_int_list_9 = None
        convert_element_type_default_10: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_default_11: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_5, torch.float32);  reshape_default_5 = None
        sum_dim_int_list_10: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_27, [0], True, dtype = torch.float32);  view_27 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [768]);  sum_dim_int_list_10 = None
        convert_element_type_default_12: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_default_13: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_6, torch.float32);  reshape_default_6 = None
        mul_tensor_2: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_7, arg119_1);  arg119_1 = None
        sum_dim_int_list_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_7, [0, 1]);  add_7 = None
        sum_dim_int_list_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_30, [0], True, dtype = torch.float32);  view_30 = None
        reshape_default_7: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, [768]);  sum_dim_int_list_13 = None
        convert_element_type_default_14: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_default_15: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_7, torch.float32);  reshape_default_7 = None
        sum_dim_int_list_14: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_34, [0], True, dtype = torch.float32);  view_34 = None
        reshape_default_8: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, [3072]);  sum_dim_int_list_14 = None
        convert_element_type_default_16: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_default_17: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_8, torch.float32);  reshape_default_8 = None
        mul_tensor_3: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_10, arg115_1);  arg115_1 = None
        sum_dim_int_list_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_10, [0, 1]);  add_10 = None
        sum_dim_int_list_17: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True, dtype = torch.float32);  view_37 = None
        reshape_default_9: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, [768]);  sum_dim_int_list_17 = None
        convert_element_type_default_18: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_default_19: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_9, torch.float32);  reshape_default_9 = None
        sum_dim_int_list_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_50, [0], True, dtype = torch.float32);  view_50 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, [768]);  sum_dim_int_list_18 = None
        convert_element_type_default_20: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_default_21: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_10, torch.float32);  reshape_default_10 = None
        sum_dim_int_list_19: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_53, [0], True, dtype = torch.float32);  view_53 = None
        reshape_default_11: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, [768]);  sum_dim_int_list_19 = None
        convert_element_type_default_22: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_default_23: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_11, torch.float32);  reshape_default_11 = None
        sum_dim_int_list_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_56, [0], True, dtype = torch.float32);  view_56 = None
        reshape_default_12: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, [768]);  sum_dim_int_list_20 = None
        convert_element_type_default_24: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_default_25: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_12, torch.float32);  reshape_default_12 = None
        mul_tensor_4: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_14, arg111_1);  arg111_1 = None
        sum_dim_int_list_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_14, [0, 1]);  add_14 = None
        sum_dim_int_list_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_59, [0], True, dtype = torch.float32);  view_59 = None
        reshape_default_13: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, [768]);  sum_dim_int_list_23 = None
        convert_element_type_default_26: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_default_27: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_13, torch.float32);  reshape_default_13 = None
        sum_dim_int_list_24: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_63, [0], True, dtype = torch.float32);  view_63 = None
        reshape_default_14: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, [3072]);  sum_dim_int_list_24 = None
        convert_element_type_default_28: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_default_29: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_14, torch.float32);  reshape_default_14 = None
        mul_tensor_5: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_17, arg107_1);  arg107_1 = None
        sum_dim_int_list_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_17, [0, 1]);  add_17 = None
        sum_dim_int_list_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_66, [0], True, dtype = torch.float32);  view_66 = None
        reshape_default_15: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, [768]);  sum_dim_int_list_27 = None
        convert_element_type_default_30: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_default_31: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_15, torch.float32);  reshape_default_15 = None
        sum_dim_int_list_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_79, [0], True, dtype = torch.float32);  view_79 = None
        reshape_default_16: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, [768]);  sum_dim_int_list_28 = None
        convert_element_type_default_32: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_default_33: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_16, torch.float32);  reshape_default_16 = None
        sum_dim_int_list_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_82, [0], True, dtype = torch.float32);  view_82 = None
        reshape_default_17: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, [768]);  sum_dim_int_list_29 = None
        convert_element_type_default_34: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_default_35: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_17, torch.float32);  reshape_default_17 = None
        sum_dim_int_list_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_85, [0], True, dtype = torch.float32);  view_85 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, [768]);  sum_dim_int_list_30 = None
        convert_element_type_default_36: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_default_37: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_18, torch.float32);  reshape_default_18 = None
        mul_tensor_6: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_21, arg103_1);  arg103_1 = None
        sum_dim_int_list_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_21, [0, 1]);  add_21 = None
        sum_dim_int_list_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_88, [0], True, dtype = torch.float32);  view_88 = None
        reshape_default_19: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, [768]);  sum_dim_int_list_33 = None
        convert_element_type_default_38: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_default_39: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_19, torch.float32);  reshape_default_19 = None
        sum_dim_int_list_34: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_92, [0], True, dtype = torch.float32);  view_92 = None
        reshape_default_20: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, [3072]);  sum_dim_int_list_34 = None
        convert_element_type_default_40: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_default_41: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_20, torch.float32);  reshape_default_20 = None
        mul_tensor_7: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_24, arg99_1);  arg99_1 = None
        sum_dim_int_list_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_24, [0, 1]);  add_24 = None
        sum_dim_int_list_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_95, [0], True, dtype = torch.float32);  view_95 = None
        reshape_default_21: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, [768]);  sum_dim_int_list_37 = None
        convert_element_type_default_42: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_default_43: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_21, torch.float32);  reshape_default_21 = None
        sum_dim_int_list_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_108, [0], True, dtype = torch.float32);  view_108 = None
        reshape_default_22: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, [768]);  sum_dim_int_list_38 = None
        convert_element_type_default_44: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_default_45: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_22, torch.float32);  reshape_default_22 = None
        sum_dim_int_list_39: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_111, [0], True, dtype = torch.float32);  view_111 = None
        reshape_default_23: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, [768]);  sum_dim_int_list_39 = None
        convert_element_type_default_46: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_default_47: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_23, torch.float32);  reshape_default_23 = None
        sum_dim_int_list_40: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True, dtype = torch.float32);  view_114 = None
        reshape_default_24: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, [768]);  sum_dim_int_list_40 = None
        convert_element_type_default_48: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_default_49: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_24, torch.float32);  reshape_default_24 = None
        mul_tensor_8: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_28, arg95_1);  arg95_1 = None
        sum_dim_int_list_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_28, [0, 1]);  add_28 = None
        sum_dim_int_list_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True, dtype = torch.float32);  view_117 = None
        reshape_default_25: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, [768]);  sum_dim_int_list_43 = None
        convert_element_type_default_50: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_default_51: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_25, torch.float32);  reshape_default_25 = None
        sum_dim_int_list_44: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_121, [0], True, dtype = torch.float32);  view_121 = None
        reshape_default_26: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, [3072]);  sum_dim_int_list_44 = None
        convert_element_type_default_52: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_default_53: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_26, torch.float32);  reshape_default_26 = None
        mul_tensor_9: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_31, arg91_1);  arg91_1 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_31, [0, 1]);  add_31 = None
        sum_dim_int_list_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_124, [0], True, dtype = torch.float32);  view_124 = None
        reshape_default_27: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, [768]);  sum_dim_int_list_47 = None
        convert_element_type_default_54: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_default_55: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_27, torch.float32);  reshape_default_27 = None
        sum_dim_int_list_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_137, [0], True, dtype = torch.float32);  view_137 = None
        reshape_default_28: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, [768]);  sum_dim_int_list_48 = None
        convert_element_type_default_56: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_default_57: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_28, torch.float32);  reshape_default_28 = None
        sum_dim_int_list_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_140, [0], True, dtype = torch.float32);  view_140 = None
        reshape_default_29: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, [768]);  sum_dim_int_list_49 = None
        convert_element_type_default_58: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_default_59: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_29, torch.float32);  reshape_default_29 = None
        sum_dim_int_list_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_143, [0], True, dtype = torch.float32);  view_143 = None
        reshape_default_30: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, [768]);  sum_dim_int_list_50 = None
        convert_element_type_default_60: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_default_61: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_30, torch.float32);  reshape_default_30 = None
        mul_tensor_10: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_35, arg87_1);  arg87_1 = None
        sum_dim_int_list_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_35, [0, 1]);  add_35 = None
        sum_dim_int_list_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_146, [0], True, dtype = torch.float32);  view_146 = None
        reshape_default_31: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, [768]);  sum_dim_int_list_53 = None
        convert_element_type_default_62: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_default_63: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_31, torch.float32);  reshape_default_31 = None
        sum_dim_int_list_54: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True, dtype = torch.float32);  view_150 = None
        reshape_default_32: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, [3072]);  sum_dim_int_list_54 = None
        convert_element_type_default_64: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_default_65: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_32, torch.float32);  reshape_default_32 = None
        mul_tensor_11: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_38, arg83_1);  arg83_1 = None
        sum_dim_int_list_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_38, [0, 1]);  add_38 = None
        sum_dim_int_list_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_153, [0], True, dtype = torch.float32);  view_153 = None
        reshape_default_33: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, [768]);  sum_dim_int_list_57 = None
        convert_element_type_default_66: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_default_67: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_33, torch.float32);  reshape_default_33 = None
        sum_dim_int_list_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_166, [0], True, dtype = torch.float32);  view_166 = None
        reshape_default_34: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, [768]);  sum_dim_int_list_58 = None
        convert_element_type_default_68: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_default_69: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_34, torch.float32);  reshape_default_34 = None
        sum_dim_int_list_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_169, [0], True, dtype = torch.float32);  view_169 = None
        reshape_default_35: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, [768]);  sum_dim_int_list_59 = None
        convert_element_type_default_70: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_default_71: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_35, torch.float32);  reshape_default_35 = None
        sum_dim_int_list_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_172, [0], True, dtype = torch.float32);  view_172 = None
        reshape_default_36: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, [768]);  sum_dim_int_list_60 = None
        convert_element_type_default_72: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_default_73: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_36, torch.float32);  reshape_default_36 = None
        mul_tensor_12: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_42, arg79_1);  arg79_1 = None
        sum_dim_int_list_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_42, [0, 1]);  add_42 = None
        sum_dim_int_list_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_175, [0], True, dtype = torch.float32);  view_175 = None
        reshape_default_37: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, [768]);  sum_dim_int_list_63 = None
        convert_element_type_default_74: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_default_75: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_37, torch.float32);  reshape_default_37 = None
        sum_dim_int_list_64: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_179, [0], True, dtype = torch.float32);  view_179 = None
        reshape_default_38: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, [3072]);  sum_dim_int_list_64 = None
        convert_element_type_default_76: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_default_77: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_38, torch.float32);  reshape_default_38 = None
        mul_tensor_13: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_45, arg75_1);  arg75_1 = None
        sum_dim_int_list_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_45, [0, 1]);  add_45 = None
        sum_dim_int_list_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True, dtype = torch.float32);  view_182 = None
        reshape_default_39: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, [768]);  sum_dim_int_list_67 = None
        convert_element_type_default_78: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_default_79: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_39, torch.float32);  reshape_default_39 = None
        sum_dim_int_list_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True, dtype = torch.float32);  view_195 = None
        reshape_default_40: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, [768]);  sum_dim_int_list_68 = None
        convert_element_type_default_80: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_default_81: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_40, torch.float32);  reshape_default_40 = None
        sum_dim_int_list_69: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True, dtype = torch.float32);  view_198 = None
        reshape_default_41: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_69, [768]);  sum_dim_int_list_69 = None
        convert_element_type_default_82: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_default_83: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_41, torch.float32);  reshape_default_41 = None
        sum_dim_int_list_70: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True, dtype = torch.float32);  view_201 = None
        reshape_default_42: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, [768]);  sum_dim_int_list_70 = None
        convert_element_type_default_84: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_default_85: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_42, torch.float32);  reshape_default_42 = None
        mul_tensor_14: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_49, arg71_1);  arg71_1 = None
        sum_dim_int_list_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_49, [0, 1]);  add_49 = None
        sum_dim_int_list_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True, dtype = torch.float32);  view_204 = None
        reshape_default_43: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, [768]);  sum_dim_int_list_73 = None
        convert_element_type_default_86: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_default_87: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_43, torch.float32);  reshape_default_43 = None
        sum_dim_int_list_74: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_208, [0], True, dtype = torch.float32);  view_208 = None
        reshape_default_44: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_74, [3072]);  sum_dim_int_list_74 = None
        convert_element_type_default_88: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_default_89: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_44, torch.float32);  reshape_default_44 = None
        mul_tensor_15: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_52, arg67_1);  arg67_1 = None
        sum_dim_int_list_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_52, [0, 1]);  add_52 = None
        sum_dim_int_list_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_211, [0], True, dtype = torch.float32);  view_211 = None
        reshape_default_45: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, [768]);  sum_dim_int_list_77 = None
        convert_element_type_default_90: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_default_91: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_45, torch.float32);  reshape_default_45 = None
        sum_dim_int_list_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True, dtype = torch.float32);  view_224 = None
        reshape_default_46: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_78, [768]);  sum_dim_int_list_78 = None
        convert_element_type_default_92: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_default_93: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_46, torch.float32);  reshape_default_46 = None
        sum_dim_int_list_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True, dtype = torch.float32);  view_227 = None
        reshape_default_47: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, [768]);  sum_dim_int_list_79 = None
        convert_element_type_default_94: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_default_95: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_47, torch.float32);  reshape_default_47 = None
        sum_dim_int_list_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_230, [0], True, dtype = torch.float32);  view_230 = None
        reshape_default_48: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, [768]);  sum_dim_int_list_80 = None
        convert_element_type_default_96: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_default_97: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_48, torch.float32);  reshape_default_48 = None
        mul_tensor_16: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_56, arg63_1);  arg63_1 = None
        sum_dim_int_list_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_56, [0, 1]);  add_56 = None
        sum_dim_int_list_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True, dtype = torch.float32);  view_233 = None
        reshape_default_49: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, [768]);  sum_dim_int_list_83 = None
        convert_element_type_default_98: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_default_99: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_49, torch.float32);  reshape_default_49 = None
        sum_dim_int_list_84: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_237, [0], True, dtype = torch.float32);  view_237 = None
        reshape_default_50: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, [3072]);  sum_dim_int_list_84 = None
        convert_element_type_default_100: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_default_101: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_50, torch.float32);  reshape_default_50 = None
        mul_tensor_17: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_59, arg59_1);  arg59_1 = None
        sum_dim_int_list_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_59, [0, 1]);  add_59 = None
        sum_dim_int_list_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True, dtype = torch.float32);  view_240 = None
        reshape_default_51: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, [768]);  sum_dim_int_list_87 = None
        convert_element_type_default_102: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_default_103: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_51, torch.float32);  reshape_default_51 = None
        sum_dim_int_list_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_253, [0], True, dtype = torch.float32);  view_253 = None
        reshape_default_52: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, [768]);  sum_dim_int_list_88 = None
        convert_element_type_default_104: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_default_105: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_52, torch.float32);  reshape_default_52 = None
        sum_dim_int_list_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_256, [0], True, dtype = torch.float32);  view_256 = None
        reshape_default_53: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_89, [768]);  sum_dim_int_list_89 = None
        convert_element_type_default_106: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_default_107: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_53, torch.float32);  reshape_default_53 = None
        sum_dim_int_list_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_259, [0], True, dtype = torch.float32);  view_259 = None
        reshape_default_54: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, [768]);  sum_dim_int_list_90 = None
        convert_element_type_default_108: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_default_109: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_54, torch.float32);  reshape_default_54 = None
        mul_tensor_18: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_63, arg55_1);  arg55_1 = None
        sum_dim_int_list_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_63, [0, 1]);  add_63 = None
        sum_dim_int_list_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_262, [0], True, dtype = torch.float32);  view_262 = None
        reshape_default_55: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, [768]);  sum_dim_int_list_93 = None
        convert_element_type_default_110: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_default_111: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_55, torch.float32);  reshape_default_55 = None
        sum_dim_int_list_94: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True, dtype = torch.float32);  view_266 = None
        reshape_default_56: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_94, [3072]);  sum_dim_int_list_94 = None
        convert_element_type_default_112: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_default_113: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_56, torch.float32);  reshape_default_56 = None
        mul_tensor_19: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_66, arg51_1);  arg51_1 = None
        sum_dim_int_list_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_66, [0, 1]);  add_66 = None
        sum_dim_int_list_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True, dtype = torch.float32);  view_269 = None
        reshape_default_57: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, [768]);  sum_dim_int_list_97 = None
        convert_element_type_default_114: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_default_115: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_57, torch.float32);  reshape_default_57 = None
        sum_dim_int_list_98: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True, dtype = torch.float32);  view_282 = None
        reshape_default_58: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_98, [768]);  sum_dim_int_list_98 = None
        convert_element_type_default_116: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_default_117: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_58, torch.float32);  reshape_default_58 = None
        sum_dim_int_list_99: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True, dtype = torch.float32);  view_285 = None
        reshape_default_59: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, [768]);  sum_dim_int_list_99 = None
        convert_element_type_default_118: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_default_119: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_59, torch.float32);  reshape_default_59 = None
        sum_dim_int_list_100: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True, dtype = torch.float32);  view_288 = None
        reshape_default_60: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, [768]);  sum_dim_int_list_100 = None
        convert_element_type_default_120: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_default_121: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_60, torch.float32);  reshape_default_60 = None
        mul_tensor_20: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_70, arg47_1);  arg47_1 = None
        sum_dim_int_list_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_70, [0, 1]);  add_70 = None
        sum_dim_int_list_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True, dtype = torch.float32);  view_291 = None
        reshape_default_61: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, [768]);  sum_dim_int_list_103 = None
        convert_element_type_default_122: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_default_123: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_61, torch.float32);  reshape_default_61 = None
        sum_dim_int_list_104: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True, dtype = torch.float32);  view_295 = None
        reshape_default_62: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, [3072]);  sum_dim_int_list_104 = None
        convert_element_type_default_124: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_default_125: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_62, torch.float32);  reshape_default_62 = None
        mul_tensor_21: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_73, arg43_1);  arg43_1 = None
        sum_dim_int_list_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_73, [0, 1]);  add_73 = None
        sum_dim_int_list_107: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True, dtype = torch.float32);  view_298 = None
        reshape_default_63: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, [768]);  sum_dim_int_list_107 = None
        convert_element_type_default_126: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_default_127: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_63, torch.float32);  reshape_default_63 = None
        sum_dim_int_list_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True, dtype = torch.float32);  view_311 = None
        reshape_default_64: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, [768]);  sum_dim_int_list_108 = None
        convert_element_type_default_128: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_default_129: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_64, torch.float32);  reshape_default_64 = None
        sum_dim_int_list_109: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True, dtype = torch.float32);  view_314 = None
        reshape_default_65: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_109, [768]);  sum_dim_int_list_109 = None
        convert_element_type_default_130: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_default_131: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_65, torch.float32);  reshape_default_65 = None
        sum_dim_int_list_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True, dtype = torch.float32);  view_317 = None
        reshape_default_66: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, [768]);  sum_dim_int_list_110 = None
        convert_element_type_default_132: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_default_133: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_66, torch.float32);  reshape_default_66 = None
        mul_tensor_22: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_77, arg39_1);  arg39_1 = None
        sum_dim_int_list_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_77, [0, 1]);  add_77 = None
        sum_dim_int_list_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True, dtype = torch.float32);  view_320 = None
        reshape_default_67: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, [768]);  sum_dim_int_list_113 = None
        convert_element_type_default_134: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_default_135: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_67, torch.float32);  reshape_default_67 = None
        sum_dim_int_list_114: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True, dtype = torch.float32);  view_324 = None
        reshape_default_68: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_114, [3072]);  sum_dim_int_list_114 = None
        convert_element_type_default_136: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_default_137: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_68, torch.float32);  reshape_default_68 = None
        mul_tensor_23: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_80, arg35_1);  arg35_1 = None
        sum_dim_int_list_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_80, [0, 1]);  add_80 = None
        add_tensor: "f32[4, 476, 768]" = torch.ops.aten.add.Tensor(arg276_1, mul_227);  arg276_1 = mul_227 = None
        sum_dim_int_list_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True, dtype = torch.float32);  view_327 = None
        reshape_default_69: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, [768]);  sum_dim_int_list_117 = None
        convert_element_type_default_138: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_default_139: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_69, torch.float32);  reshape_default_69 = None
        sum_dim_int_list_118: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True, dtype = torch.float32);  view_341 = None
        reshape_default_70: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_118, [768]);  sum_dim_int_list_118 = None
        reshape_default_71: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(mm_140, [4, 476, 768]);  mm_140 = None
        convert_element_type_default_140: "f32[4, 476, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_71, torch.float32);  reshape_default_71 = None
        add_tensor_1: "f32[4, 476, 768]" = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_140);  add_tensor = convert_element_type_default_140 = None
        convert_element_type_default_141: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_default_142: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_70, torch.float32);  reshape_default_70 = None
        sum_dim_int_list_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True, dtype = torch.float32);  view_344 = None
        reshape_default_72: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_119, [768]);  sum_dim_int_list_119 = None
        reshape_default_73: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(mm_142, [4, 476, 768]);  mm_142 = None
        convert_element_type_default_143: "f32[4, 476, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_73, torch.float32);  reshape_default_73 = None
        add_tensor_2: "f32[4, 476, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_143);  add_tensor_1 = convert_element_type_default_143 = None
        convert_element_type_default_144: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_default_145: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_72, torch.float32);  reshape_default_72 = None
        sum_dim_int_list_120: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True, dtype = torch.float32);  view_347 = None
        reshape_default_74: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, [768]);  sum_dim_int_list_120 = None
        reshape_default_75: "f16[4, 476, 768]" = torch.ops.aten.reshape.default(mm_144, [4, 476, 768]);  mm_144 = None
        convert_element_type_default_146: "f32[4, 476, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_75, torch.float32);  reshape_default_75 = None
        add_tensor_3: "f32[4, 476, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_146);  add_tensor_2 = convert_element_type_default_146 = None
        convert_element_type_default_147: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_default_148: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_74, torch.float32);  reshape_default_74 = None
        mul_tensor_24: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_tensor_3, arg2_1);  arg2_1 = None
        mul_tensor_25: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_24, 768)
        sum_dim_int_list_121: "f32[4, 476, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [2], True)
        mul_tensor_26: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_24, arg29_1);  mul_tensor_24 = None
        sum_dim_int_list_122: "f32[4, 476, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [2], True);  mul_tensor_26 = None
        mul_tensor_27: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(arg29_1, sum_dim_int_list_122);  sum_dim_int_list_122 = None
        sub_tensor: "f32[4, 476, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_25, sum_dim_int_list_121);  mul_tensor_25 = sum_dim_int_list_121 = None
        sub_tensor_1: "f32[4, 476, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_27);  sub_tensor = mul_tensor_27 = None
        mul_tensor_28: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(arg275_1, sub_tensor_1);  arg275_1 = sub_tensor_1 = None
        mul_tensor_29: "f32[4, 476, 768]" = torch.ops.aten.mul.Tensor(add_tensor_3, arg29_1);  arg29_1 = None
        sum_dim_int_list_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1]);  mul_tensor_29 = None
        sum_dim_int_list_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_tensor_3, [0, 1]);  add_tensor_3 = None
        eq_scalar: "b8[4, 476]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default: "b8[4, 476, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 476, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, mul_tensor_28);  unsqueeze_default = None
        full_default_1: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2, 768]" = torch.ops.aten.index_put.default(full_default_1, [arg0_1], where_self, True);  full_default_1 = arg0_1 = where_self = None
        expand_default: "i64[4, 476]" = torch.ops.aten.expand.default(arg28_1, [4, 476]);  arg28_1 = None
        eq_scalar_1: "b8[4, 476]" = torch.ops.aten.eq.Scalar(expand_default, -1)
        unsqueeze_default_1: "b8[4, 476, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[4, 476, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default, mul_tensor_28);  unsqueeze_default_1 = None
        full_default_2: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default_2, [expand_default], where_self_1, True);  full_default_2 = expand_default = where_self_1 = None
        eq_scalar_2: "b8[4, 476]" = torch.ops.aten.eq.Scalar(arg1_1, 0)
        unsqueeze_default_2: "b8[4, 476, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[4, 476, 768]" = torch.ops.aten.where.self(unsqueeze_default_2, full_default, mul_tensor_28);  unsqueeze_default_2 = full_default = mul_tensor_28 = None
        full_default_3: "f32[21128, 768]" = torch.ops.aten.full.default([21128, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_2: "f32[21128, 768]" = torch.ops.aten.index_put.default(full_default_3, [arg1_1], where_self_2, True);  full_default_3 = arg1_1 = where_self_2 = None
        return (convert_element_type_default, convert_element_type_default_1, sum_dim_int_list_1, sum_dim_int_list_2, convert_element_type_default_2, convert_element_type_default_3, convert_element_type_default_4, convert_element_type_default_5, sum_dim_int_list_5, sum_dim_int_list_6, convert_element_type_default_6, convert_element_type_default_7, convert_element_type_default_8, convert_element_type_default_9, convert_element_type_default_10, convert_element_type_default_11, convert_element_type_default_12, convert_element_type_default_13, sum_dim_int_list_11, sum_dim_int_list_12, convert_element_type_default_14, convert_element_type_default_15, convert_element_type_default_16, convert_element_type_default_17, sum_dim_int_list_15, sum_dim_int_list_16, convert_element_type_default_18, convert_element_type_default_19, convert_element_type_default_20, convert_element_type_default_21, convert_element_type_default_22, convert_element_type_default_23, convert_element_type_default_24, convert_element_type_default_25, sum_dim_int_list_21, sum_dim_int_list_22, convert_element_type_default_26, convert_element_type_default_27, convert_element_type_default_28, convert_element_type_default_29, sum_dim_int_list_25, sum_dim_int_list_26, convert_element_type_default_30, convert_element_type_default_31, convert_element_type_default_32, convert_element_type_default_33, convert_element_type_default_34, convert_element_type_default_35, convert_element_type_default_36, convert_element_type_default_37, sum_dim_int_list_31, sum_dim_int_list_32, convert_element_type_default_38, convert_element_type_default_39, convert_element_type_default_40, convert_element_type_default_41, sum_dim_int_list_35, sum_dim_int_list_36, convert_element_type_default_42, convert_element_type_default_43, convert_element_type_default_44, convert_element_type_default_45, convert_element_type_default_46, convert_element_type_default_47, convert_element_type_default_48, convert_element_type_default_49, sum_dim_int_list_41, sum_dim_int_list_42, convert_element_type_default_50, convert_element_type_default_51, convert_element_type_default_52, convert_element_type_default_53, sum_dim_int_list_45, sum_dim_int_list_46, convert_element_type_default_54, convert_element_type_default_55, convert_element_type_default_56, convert_element_type_default_57, convert_element_type_default_58, convert_element_type_default_59, convert_element_type_default_60, convert_element_type_default_61, sum_dim_int_list_51, sum_dim_int_list_52, convert_element_type_default_62, convert_element_type_default_63, convert_element_type_default_64, convert_element_type_default_65, sum_dim_int_list_55, sum_dim_int_list_56, convert_element_type_default_66, convert_element_type_default_67, convert_element_type_default_68, convert_element_type_default_69, convert_element_type_default_70, convert_element_type_default_71, convert_element_type_default_72, convert_element_type_default_73, sum_dim_int_list_61, sum_dim_int_list_62, convert_element_type_default_74, convert_element_type_default_75, convert_element_type_default_76, convert_element_type_default_77, sum_dim_int_list_65, sum_dim_int_list_66, convert_element_type_default_78, convert_element_type_default_79, convert_element_type_default_80, convert_element_type_default_81, convert_element_type_default_82, convert_element_type_default_83, convert_element_type_default_84, convert_element_type_default_85, sum_dim_int_list_71, sum_dim_int_list_72, convert_element_type_default_86, convert_element_type_default_87, convert_element_type_default_88, convert_element_type_default_89, sum_dim_int_list_75, sum_dim_int_list_76, convert_element_type_default_90, convert_element_type_default_91, convert_element_type_default_92, convert_element_type_default_93, convert_element_type_default_94, convert_element_type_default_95, convert_element_type_default_96, convert_element_type_default_97, sum_dim_int_list_81, sum_dim_int_list_82, convert_element_type_default_98, convert_element_type_default_99, convert_element_type_default_100, convert_element_type_default_101, sum_dim_int_list_85, sum_dim_int_list_86, convert_element_type_default_102, convert_element_type_default_103, convert_element_type_default_104, convert_element_type_default_105, convert_element_type_default_106, convert_element_type_default_107, convert_element_type_default_108, convert_element_type_default_109, sum_dim_int_list_91, sum_dim_int_list_92, convert_element_type_default_110, convert_element_type_default_111, convert_element_type_default_112, convert_element_type_default_113, sum_dim_int_list_95, sum_dim_int_list_96, convert_element_type_default_114, convert_element_type_default_115, convert_element_type_default_116, convert_element_type_default_117, convert_element_type_default_118, convert_element_type_default_119, convert_element_type_default_120, convert_element_type_default_121, sum_dim_int_list_101, sum_dim_int_list_102, convert_element_type_default_122, convert_element_type_default_123, convert_element_type_default_124, convert_element_type_default_125, sum_dim_int_list_105, sum_dim_int_list_106, convert_element_type_default_126, convert_element_type_default_127, convert_element_type_default_128, convert_element_type_default_129, convert_element_type_default_130, convert_element_type_default_131, convert_element_type_default_132, convert_element_type_default_133, sum_dim_int_list_111, sum_dim_int_list_112, convert_element_type_default_134, convert_element_type_default_135, convert_element_type_default_136, convert_element_type_default_137, sum_dim_int_list_115, sum_dim_int_list_116, convert_element_type_default_138, convert_element_type_default_139, convert_element_type_default_141, convert_element_type_default_142, convert_element_type_default_144, convert_element_type_default_145, convert_element_type_default_147, convert_element_type_default_148, sum_dim_int_list_123, sum_dim_int_list_124, index_put_default, index_put_default_1, index_put_default_2)


def _default_make_inputs():
    return [
    torch.randn([4, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1904, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 476, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 476], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 476], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [4, 476], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
