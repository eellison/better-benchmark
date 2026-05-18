"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g36
Pattern hash: fb2a28eb81be
Shape hash: e396637d
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
    def forward(self, convert_element_type_4: "f16[512, 30522]", mm_1: "f16[30522, 768]", convert_element_type_5: "f32[1, 512, 768]", mul_7: "f32[1, 512, 768]", view_5: "f16[512, 768]", mm_3: "f16[768, 768]", convert_element_type_12: "f32[1, 512, 768]", arg193_1: "f32[1, 512, 768]", view_8: "f16[512, 768]", mm_5: "f16[768, 3072]", view_12: "f16[512, 3072]", mm_7: "f16[3072, 768]", add_4: "f32[1, 512, 768]", arg188_1: "f32[1, 512, 768]", view_15: "f16[512, 768]", mm_9: "f16[768, 768]", view_27: "f16[512, 768]", mm_11: "f16[768, 768]", view_31: "f16[512, 768]", mm_13: "f16[768, 768]", view_35: "f16[512, 768]", mm_15: "f16[768, 768]", add_7: "f32[1, 512, 768]", arg180_1: "f32[1, 512, 768]", view_38: "f16[512, 768]", mm_17: "f16[768, 3072]", view_42: "f16[512, 3072]", mm_19: "f16[3072, 768]", add_10: "f32[1, 512, 768]", arg175_1: "f32[1, 512, 768]", view_45: "f16[512, 768]", mm_21: "f16[768, 768]", view_57: "f16[512, 768]", mm_23: "f16[768, 768]", view_61: "f16[512, 768]", mm_25: "f16[768, 768]", view_65: "f16[512, 768]", mm_27: "f16[768, 768]", add_13: "f32[1, 512, 768]", arg167_1: "f32[1, 512, 768]", view_68: "f16[512, 768]", mm_29: "f16[768, 3072]", view_72: "f16[512, 3072]", mm_31: "f16[3072, 768]", add_16: "f32[1, 512, 768]", arg162_1: "f32[1, 512, 768]", view_75: "f16[512, 768]", mm_33: "f16[768, 768]", view_87: "f16[512, 768]", mm_35: "f16[768, 768]", view_91: "f16[512, 768]", mm_37: "f16[768, 768]", view_95: "f16[512, 768]", mm_39: "f16[768, 768]", add_19: "f32[1, 512, 768]", arg154_1: "f32[1, 512, 768]", view_98: "f16[512, 768]", mm_41: "f16[768, 3072]", view_102: "f16[512, 3072]", mm_43: "f16[3072, 768]", add_22: "f32[1, 512, 768]", arg149_1: "f32[1, 512, 768]", view_105: "f16[512, 768]", mm_45: "f16[768, 768]", view_117: "f16[512, 768]", mm_47: "f16[768, 768]", view_121: "f16[512, 768]", mm_49: "f16[768, 768]", view_125: "f16[512, 768]", mm_51: "f16[768, 768]", add_25: "f32[1, 512, 768]", arg141_1: "f32[1, 512, 768]", view_128: "f16[512, 768]", mm_53: "f16[768, 3072]", view_132: "f16[512, 3072]", mm_55: "f16[3072, 768]", add_28: "f32[1, 512, 768]", arg136_1: "f32[1, 512, 768]", view_135: "f16[512, 768]", mm_57: "f16[768, 768]", view_147: "f16[512, 768]", mm_59: "f16[768, 768]", view_151: "f16[512, 768]", mm_61: "f16[768, 768]", view_155: "f16[512, 768]", mm_63: "f16[768, 768]", add_31: "f32[1, 512, 768]", arg128_1: "f32[1, 512, 768]", view_158: "f16[512, 768]", mm_65: "f16[768, 3072]", view_162: "f16[512, 3072]", mm_67: "f16[3072, 768]", add_34: "f32[1, 512, 768]", arg123_1: "f32[1, 512, 768]", view_165: "f16[512, 768]", mm_69: "f16[768, 768]", view_177: "f16[512, 768]", mm_71: "f16[768, 768]", view_181: "f16[512, 768]", mm_73: "f16[768, 768]", view_185: "f16[512, 768]", mm_75: "f16[768, 768]", add_37: "f32[1, 512, 768]", arg115_1: "f32[1, 512, 768]", view_188: "f16[512, 768]", mm_77: "f16[768, 3072]", view_192: "f16[512, 3072]", mm_79: "f16[3072, 768]", add_40: "f32[1, 512, 768]", arg110_1: "f32[1, 512, 768]", view_195: "f16[512, 768]", mm_81: "f16[768, 768]", view_207: "f16[512, 768]", mm_83: "f16[768, 768]", view_211: "f16[512, 768]", mm_85: "f16[768, 768]", view_215: "f16[512, 768]", mm_87: "f16[768, 768]", add_43: "f32[1, 512, 768]", arg102_1: "f32[1, 512, 768]", view_218: "f16[512, 768]", mm_89: "f16[768, 3072]", view_222: "f16[512, 3072]", mm_91: "f16[3072, 768]", add_46: "f32[1, 512, 768]", arg97_1: "f32[1, 512, 768]", view_225: "f16[512, 768]", mm_93: "f16[768, 768]", view_237: "f16[512, 768]", mm_95: "f16[768, 768]", view_241: "f16[512, 768]", mm_97: "f16[768, 768]", view_245: "f16[512, 768]", mm_99: "f16[768, 768]", add_49: "f32[1, 512, 768]", arg89_1: "f32[1, 512, 768]", view_248: "f16[512, 768]", mm_101: "f16[768, 3072]", view_252: "f16[512, 3072]", mm_103: "f16[3072, 768]", add_52: "f32[1, 512, 768]", arg84_1: "f32[1, 512, 768]", view_255: "f16[512, 768]", mm_105: "f16[768, 768]", view_267: "f16[512, 768]", mm_107: "f16[768, 768]", view_271: "f16[512, 768]", mm_109: "f16[768, 768]", view_275: "f16[512, 768]", mm_111: "f16[768, 768]", add_55: "f32[1, 512, 768]", arg76_1: "f32[1, 512, 768]", view_278: "f16[512, 768]", mm_113: "f16[768, 3072]", view_282: "f16[512, 3072]", mm_115: "f16[3072, 768]", add_58: "f32[1, 512, 768]", arg71_1: "f32[1, 512, 768]", view_285: "f16[512, 768]", mm_117: "f16[768, 768]", view_297: "f16[512, 768]", mm_119: "f16[768, 768]", view_301: "f16[512, 768]", mm_121: "f16[768, 768]", view_305: "f16[512, 768]", mm_123: "f16[768, 768]", add_61: "f32[1, 512, 768]", arg63_1: "f32[1, 512, 768]", view_308: "f16[512, 768]", mm_125: "f16[768, 3072]", view_312: "f16[512, 3072]", mm_127: "f16[3072, 768]", add_64: "f32[1, 512, 768]", arg58_1: "f32[1, 512, 768]", view_315: "f16[512, 768]", mm_129: "f16[768, 768]", view_327: "f16[512, 768]", mm_131: "f16[768, 768]", view_331: "f16[512, 768]", mm_133: "f16[768, 768]", view_335: "f16[512, 768]", mm_135: "f16[768, 768]", add_67: "f32[1, 512, 768]", arg50_1: "f32[1, 512, 768]", view_338: "f16[512, 768]", mm_137: "f16[768, 3072]", view_342: "f16[512, 3072]", mm_139: "f16[3072, 768]", add_70: "f32[1, 512, 768]", arg45_1: "f32[1, 512, 768]", view_345: "f16[512, 768]", mm_141: "f16[768, 768]", view_357: "f16[512, 768]", mm_142: "f16[512, 768]", mul_356: "f32[1, 512, 768]", mm_143: "f16[768, 768]", view_361: "f16[512, 768]", mm_144: "f16[512, 768]", mm_145: "f16[768, 768]", view_365: "f16[512, 768]", mm_146: "f16[512, 768]", mm_147: "f16[768, 768]", arg37_1: "b8[1, 512, 768]", arg2_1: "f32[768]", arg36_1: "f32[1, 512, 768]", arg349_1: "f32[1, 512, 1]", full_1: "f32[]", arg29_1: "i64[1, 512]", arg35_1: "i64[1, 512]", arg34_1: "i64[1, 512]", arg33_1: "i64[1, 512]", arg32_1: "i64[1, 512]", arg31_1: "i64[1, 512]", arg30_1: "i64[1, 512]", arg1_1: "i64[1, 512]", arg0_1: "i64[1, 512]"):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f16[1, 30522]" = torch.ops.aten.sum.dim_IntList(convert_element_type_4, [0], True);  convert_element_type_4 = None
        reshape_default: "f16[30522]" = torch.ops.aten.reshape.default(sum_dim_int_list, [30522]);  sum_dim_int_list = None
        convert_element_type_default: "f32[30522, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_default_1: "f32[30522]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, mul_7);  mul_7 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_5, [0, 1]);  convert_element_type_5 = None
        sum_dim_int_list_3: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_5, [0], True);  view_5 = None
        reshape_default_1: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [768]);  sum_dim_int_list_3 = None
        convert_element_type_default_2: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_default_3: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, arg193_1);  arg193_1 = None
        sum_dim_int_list_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_12, [0, 1]);  convert_element_type_12 = None
        sum_dim_int_list_6: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True);  view_8 = None
        reshape_default_2: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, [768]);  sum_dim_int_list_6 = None
        convert_element_type_default_4: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_default_5: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        sum_dim_int_list_7: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_12, [0], True);  view_12 = None
        reshape_default_3: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [3072]);  sum_dim_int_list_7 = None
        convert_element_type_default_6: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_default_7: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_3, torch.float32);  reshape_default_3 = None
        mul_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_4, arg188_1);  arg188_1 = None
        sum_dim_int_list_8: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_4, [0, 1]);  add_4 = None
        sum_dim_int_list_10: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True);  view_15 = None
        reshape_default_4: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [768]);  sum_dim_int_list_10 = None
        convert_element_type_default_8: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_default_9: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_4, torch.float32);  reshape_default_4 = None
        sum_dim_int_list_11: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_27, [0], True);  view_27 = None
        reshape_default_5: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [768]);  sum_dim_int_list_11 = None
        convert_element_type_default_10: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_default_11: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_5, torch.float32);  reshape_default_5 = None
        sum_dim_int_list_12: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_31, [0], True);  view_31 = None
        reshape_default_6: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, [768]);  sum_dim_int_list_12 = None
        convert_element_type_default_12: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_default_13: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_6, torch.float32);  reshape_default_6 = None
        sum_dim_int_list_13: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_35, [0], True);  view_35 = None
        reshape_default_7: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, [768]);  sum_dim_int_list_13 = None
        convert_element_type_default_14: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_default_15: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_7, torch.float32);  reshape_default_7 = None
        mul_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_7, arg180_1);  arg180_1 = None
        sum_dim_int_list_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_7, [0, 1]);  add_7 = None
        sum_dim_int_list_16: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_38, [0], True);  view_38 = None
        reshape_default_8: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, [768]);  sum_dim_int_list_16 = None
        convert_element_type_default_16: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_default_17: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_8, torch.float32);  reshape_default_8 = None
        sum_dim_int_list_17: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_42, [0], True);  view_42 = None
        reshape_default_9: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, [3072]);  sum_dim_int_list_17 = None
        convert_element_type_default_18: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_default_19: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_9, torch.float32);  reshape_default_9 = None
        mul_tensor_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_10, arg175_1);  arg175_1 = None
        sum_dim_int_list_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_10, [0, 1]);  add_10 = None
        sum_dim_int_list_20: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_45, [0], True);  view_45 = None
        reshape_default_10: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, [768]);  sum_dim_int_list_20 = None
        convert_element_type_default_20: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_default_21: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_10, torch.float32);  reshape_default_10 = None
        sum_dim_int_list_21: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_57, [0], True);  view_57 = None
        reshape_default_11: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, [768]);  sum_dim_int_list_21 = None
        convert_element_type_default_22: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_default_23: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_11, torch.float32);  reshape_default_11 = None
        sum_dim_int_list_22: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_61, [0], True);  view_61 = None
        reshape_default_12: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, [768]);  sum_dim_int_list_22 = None
        convert_element_type_default_24: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_default_25: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_12, torch.float32);  reshape_default_12 = None
        sum_dim_int_list_23: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_65, [0], True);  view_65 = None
        reshape_default_13: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, [768]);  sum_dim_int_list_23 = None
        convert_element_type_default_26: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_default_27: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_13, torch.float32);  reshape_default_13 = None
        mul_tensor_5: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_13, arg167_1);  arg167_1 = None
        sum_dim_int_list_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_13, [0, 1]);  add_13 = None
        sum_dim_int_list_26: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_68, [0], True);  view_68 = None
        reshape_default_14: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, [768]);  sum_dim_int_list_26 = None
        convert_element_type_default_28: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_default_29: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_14, torch.float32);  reshape_default_14 = None
        sum_dim_int_list_27: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_72, [0], True);  view_72 = None
        reshape_default_15: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, [3072]);  sum_dim_int_list_27 = None
        convert_element_type_default_30: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_default_31: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_15, torch.float32);  reshape_default_15 = None
        mul_tensor_6: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_16, arg162_1);  arg162_1 = None
        sum_dim_int_list_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_16, [0, 1]);  add_16 = None
        sum_dim_int_list_30: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True);  view_75 = None
        reshape_default_16: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, [768]);  sum_dim_int_list_30 = None
        convert_element_type_default_32: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_default_33: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_16, torch.float32);  reshape_default_16 = None
        sum_dim_int_list_31: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_87, [0], True);  view_87 = None
        reshape_default_17: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, [768]);  sum_dim_int_list_31 = None
        convert_element_type_default_34: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_default_35: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_17, torch.float32);  reshape_default_17 = None
        sum_dim_int_list_32: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_91, [0], True);  view_91 = None
        reshape_default_18: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, [768]);  sum_dim_int_list_32 = None
        convert_element_type_default_36: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_default_37: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_18, torch.float32);  reshape_default_18 = None
        sum_dim_int_list_33: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_95, [0], True);  view_95 = None
        reshape_default_19: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, [768]);  sum_dim_int_list_33 = None
        convert_element_type_default_38: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_default_39: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_19, torch.float32);  reshape_default_19 = None
        mul_tensor_7: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_19, arg154_1);  arg154_1 = None
        sum_dim_int_list_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_19, [0, 1]);  add_19 = None
        sum_dim_int_list_36: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True);  view_98 = None
        reshape_default_20: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, [768]);  sum_dim_int_list_36 = None
        convert_element_type_default_40: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_default_41: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_20, torch.float32);  reshape_default_20 = None
        sum_dim_int_list_37: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_102, [0], True);  view_102 = None
        reshape_default_21: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, [3072]);  sum_dim_int_list_37 = None
        convert_element_type_default_42: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_default_43: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_21, torch.float32);  reshape_default_21 = None
        mul_tensor_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_22, arg149_1);  arg149_1 = None
        sum_dim_int_list_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_22, [0, 1]);  add_22 = None
        sum_dim_int_list_40: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True);  view_105 = None
        reshape_default_22: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, [768]);  sum_dim_int_list_40 = None
        convert_element_type_default_44: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_default_45: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_22, torch.float32);  reshape_default_22 = None
        sum_dim_int_list_41: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True);  view_117 = None
        reshape_default_23: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, [768]);  sum_dim_int_list_41 = None
        convert_element_type_default_46: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_default_47: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_23, torch.float32);  reshape_default_23 = None
        sum_dim_int_list_42: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_121, [0], True);  view_121 = None
        reshape_default_24: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, [768]);  sum_dim_int_list_42 = None
        convert_element_type_default_48: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_default_49: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_24, torch.float32);  reshape_default_24 = None
        sum_dim_int_list_43: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_125, [0], True);  view_125 = None
        reshape_default_25: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, [768]);  sum_dim_int_list_43 = None
        convert_element_type_default_50: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_default_51: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_25, torch.float32);  reshape_default_25 = None
        mul_tensor_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_25, arg141_1);  arg141_1 = None
        sum_dim_int_list_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_25, [0, 1]);  add_25 = None
        sum_dim_int_list_46: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_128, [0], True);  view_128 = None
        reshape_default_26: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, [768]);  sum_dim_int_list_46 = None
        convert_element_type_default_52: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_default_53: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_26, torch.float32);  reshape_default_26 = None
        sum_dim_int_list_47: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_132, [0], True);  view_132 = None
        reshape_default_27: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, [3072]);  sum_dim_int_list_47 = None
        convert_element_type_default_54: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_default_55: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_27, torch.float32);  reshape_default_27 = None
        mul_tensor_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_28, arg136_1);  arg136_1 = None
        sum_dim_int_list_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_28, [0, 1]);  add_28 = None
        sum_dim_int_list_50: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
        reshape_default_28: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, [768]);  sum_dim_int_list_50 = None
        convert_element_type_default_56: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_default_57: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_28, torch.float32);  reshape_default_28 = None
        sum_dim_int_list_51: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_147, [0], True);  view_147 = None
        reshape_default_29: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, [768]);  sum_dim_int_list_51 = None
        convert_element_type_default_58: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_default_59: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_29, torch.float32);  reshape_default_29 = None
        sum_dim_int_list_52: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_151, [0], True);  view_151 = None
        reshape_default_30: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, [768]);  sum_dim_int_list_52 = None
        convert_element_type_default_60: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_default_61: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_30, torch.float32);  reshape_default_30 = None
        sum_dim_int_list_53: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_155, [0], True);  view_155 = None
        reshape_default_31: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, [768]);  sum_dim_int_list_53 = None
        convert_element_type_default_62: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_default_63: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_31, torch.float32);  reshape_default_31 = None
        mul_tensor_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_31, arg128_1);  arg128_1 = None
        sum_dim_int_list_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_31, [0, 1]);  add_31 = None
        sum_dim_int_list_56: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_158, [0], True);  view_158 = None
        reshape_default_32: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, [768]);  sum_dim_int_list_56 = None
        convert_element_type_default_64: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_default_65: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_32, torch.float32);  reshape_default_32 = None
        sum_dim_int_list_57: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_162, [0], True);  view_162 = None
        reshape_default_33: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, [3072]);  sum_dim_int_list_57 = None
        convert_element_type_default_66: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_default_67: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_33, torch.float32);  reshape_default_33 = None
        mul_tensor_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_34, arg123_1);  arg123_1 = None
        sum_dim_int_list_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_34, [0, 1]);  add_34 = None
        sum_dim_int_list_60: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
        reshape_default_34: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, [768]);  sum_dim_int_list_60 = None
        convert_element_type_default_68: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_default_69: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_34, torch.float32);  reshape_default_34 = None
        sum_dim_int_list_61: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_177, [0], True);  view_177 = None
        reshape_default_35: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, [768]);  sum_dim_int_list_61 = None
        convert_element_type_default_70: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_default_71: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_35, torch.float32);  reshape_default_35 = None
        sum_dim_int_list_62: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_181, [0], True);  view_181 = None
        reshape_default_36: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, [768]);  sum_dim_int_list_62 = None
        convert_element_type_default_72: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_default_73: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_36, torch.float32);  reshape_default_36 = None
        sum_dim_int_list_63: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
        reshape_default_37: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, [768]);  sum_dim_int_list_63 = None
        convert_element_type_default_74: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_default_75: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_37, torch.float32);  reshape_default_37 = None
        mul_tensor_13: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_37, arg115_1);  arg115_1 = None
        sum_dim_int_list_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_37, [0, 1]);  add_37 = None
        sum_dim_int_list_66: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
        reshape_default_38: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, [768]);  sum_dim_int_list_66 = None
        convert_element_type_default_76: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_default_77: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_38, torch.float32);  reshape_default_38 = None
        sum_dim_int_list_67: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True);  view_192 = None
        reshape_default_39: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, [3072]);  sum_dim_int_list_67 = None
        convert_element_type_default_78: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_default_79: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_39, torch.float32);  reshape_default_39 = None
        mul_tensor_14: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_40, arg110_1);  arg110_1 = None
        sum_dim_int_list_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_40, [0, 1]);  add_40 = None
        sum_dim_int_list_70: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True);  view_195 = None
        reshape_default_40: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, [768]);  sum_dim_int_list_70 = None
        convert_element_type_default_80: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_default_81: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_40, torch.float32);  reshape_default_40 = None
        sum_dim_int_list_71: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
        reshape_default_41: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, [768]);  sum_dim_int_list_71 = None
        convert_element_type_default_82: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_default_83: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_41, torch.float32);  reshape_default_41 = None
        sum_dim_int_list_72: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_211, [0], True);  view_211 = None
        reshape_default_42: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, [768]);  sum_dim_int_list_72 = None
        convert_element_type_default_84: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_default_85: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_42, torch.float32);  reshape_default_42 = None
        sum_dim_int_list_73: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
        reshape_default_43: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, [768]);  sum_dim_int_list_73 = None
        convert_element_type_default_86: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_default_87: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_43, torch.float32);  reshape_default_43 = None
        mul_tensor_15: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_43, arg102_1);  arg102_1 = None
        sum_dim_int_list_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_43, [0, 1]);  add_43 = None
        sum_dim_int_list_76: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
        reshape_default_44: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, [768]);  sum_dim_int_list_76 = None
        convert_element_type_default_88: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_default_89: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_44, torch.float32);  reshape_default_44 = None
        sum_dim_int_list_77: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
        reshape_default_45: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, [3072]);  sum_dim_int_list_77 = None
        convert_element_type_default_90: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_default_91: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_45, torch.float32);  reshape_default_45 = None
        mul_tensor_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_46, arg97_1);  arg97_1 = None
        sum_dim_int_list_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_46, [0, 1]);  add_46 = None
        sum_dim_int_list_80: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_225, [0], True);  view_225 = None
        reshape_default_46: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, [768]);  sum_dim_int_list_80 = None
        convert_element_type_default_92: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_default_93: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_46, torch.float32);  reshape_default_46 = None
        sum_dim_int_list_81: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_237, [0], True);  view_237 = None
        reshape_default_47: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, [768]);  sum_dim_int_list_81 = None
        convert_element_type_default_94: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_default_95: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_47, torch.float32);  reshape_default_47 = None
        sum_dim_int_list_82: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_241, [0], True);  view_241 = None
        reshape_default_48: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, [768]);  sum_dim_int_list_82 = None
        convert_element_type_default_96: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_default_97: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_48, torch.float32);  reshape_default_48 = None
        sum_dim_int_list_83: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_245, [0], True);  view_245 = None
        reshape_default_49: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, [768]);  sum_dim_int_list_83 = None
        convert_element_type_default_98: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_default_99: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_49, torch.float32);  reshape_default_49 = None
        mul_tensor_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_49, arg89_1);  arg89_1 = None
        sum_dim_int_list_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_49, [0, 1]);  add_49 = None
        sum_dim_int_list_86: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
        reshape_default_50: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, [768]);  sum_dim_int_list_86 = None
        convert_element_type_default_100: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_default_101: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_50, torch.float32);  reshape_default_50 = None
        sum_dim_int_list_87: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_252, [0], True);  view_252 = None
        reshape_default_51: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, [3072]);  sum_dim_int_list_87 = None
        convert_element_type_default_102: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_default_103: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_51, torch.float32);  reshape_default_51 = None
        mul_tensor_18: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_52, arg84_1);  arg84_1 = None
        sum_dim_int_list_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_52, [0, 1]);  add_52 = None
        sum_dim_int_list_90: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_255, [0], True);  view_255 = None
        reshape_default_52: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, [768]);  sum_dim_int_list_90 = None
        convert_element_type_default_104: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_default_105: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_52, torch.float32);  reshape_default_52 = None
        sum_dim_int_list_91: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_267, [0], True);  view_267 = None
        reshape_default_53: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, [768]);  sum_dim_int_list_91 = None
        convert_element_type_default_106: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_default_107: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_53, torch.float32);  reshape_default_53 = None
        sum_dim_int_list_92: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
        reshape_default_54: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, [768]);  sum_dim_int_list_92 = None
        convert_element_type_default_108: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_default_109: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_54, torch.float32);  reshape_default_54 = None
        sum_dim_int_list_93: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
        reshape_default_55: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, [768]);  sum_dim_int_list_93 = None
        convert_element_type_default_110: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_default_111: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_55, torch.float32);  reshape_default_55 = None
        mul_tensor_19: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_55, arg76_1);  arg76_1 = None
        sum_dim_int_list_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_55, [0, 1]);  add_55 = None
        sum_dim_int_list_96: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
        reshape_default_56: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, [768]);  sum_dim_int_list_96 = None
        convert_element_type_default_112: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_default_113: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_56, torch.float32);  reshape_default_56 = None
        sum_dim_int_list_97: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
        reshape_default_57: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, [3072]);  sum_dim_int_list_97 = None
        convert_element_type_default_114: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_default_115: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_57, torch.float32);  reshape_default_57 = None
        mul_tensor_20: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_58, arg71_1);  arg71_1 = None
        sum_dim_int_list_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_58, [0, 1]);  add_58 = None
        sum_dim_int_list_100: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        reshape_default_58: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, [768]);  sum_dim_int_list_100 = None
        convert_element_type_default_116: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_default_117: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_58, torch.float32);  reshape_default_58 = None
        sum_dim_int_list_101: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_297, [0], True);  view_297 = None
        reshape_default_59: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, [768]);  sum_dim_int_list_101 = None
        convert_element_type_default_118: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_default_119: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_59, torch.float32);  reshape_default_59 = None
        sum_dim_int_list_102: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True);  view_301 = None
        reshape_default_60: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, [768]);  sum_dim_int_list_102 = None
        convert_element_type_default_120: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_default_121: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_60, torch.float32);  reshape_default_60 = None
        sum_dim_int_list_103: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
        reshape_default_61: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, [768]);  sum_dim_int_list_103 = None
        convert_element_type_default_122: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_default_123: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_61, torch.float32);  reshape_default_61 = None
        mul_tensor_21: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_61, arg63_1);  arg63_1 = None
        sum_dim_int_list_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_61, [0, 1]);  add_61 = None
        sum_dim_int_list_106: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
        reshape_default_62: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, [768]);  sum_dim_int_list_106 = None
        convert_element_type_default_124: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_default_125: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_62, torch.float32);  reshape_default_62 = None
        sum_dim_int_list_107: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_312, [0], True);  view_312 = None
        reshape_default_63: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, [3072]);  sum_dim_int_list_107 = None
        convert_element_type_default_126: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_default_127: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_63, torch.float32);  reshape_default_63 = None
        mul_tensor_22: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_64, arg58_1);  arg58_1 = None
        sum_dim_int_list_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_64, [0, 1]);  add_64 = None
        sum_dim_int_list_110: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
        reshape_default_64: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, [768]);  sum_dim_int_list_110 = None
        convert_element_type_default_128: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_default_129: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_64, torch.float32);  reshape_default_64 = None
        sum_dim_int_list_111: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
        reshape_default_65: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, [768]);  sum_dim_int_list_111 = None
        convert_element_type_default_130: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_default_131: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_65, torch.float32);  reshape_default_65 = None
        sum_dim_int_list_112: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
        reshape_default_66: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, [768]);  sum_dim_int_list_112 = None
        convert_element_type_default_132: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_default_133: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_66, torch.float32);  reshape_default_66 = None
        sum_dim_int_list_113: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
        reshape_default_67: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, [768]);  sum_dim_int_list_113 = None
        convert_element_type_default_134: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_default_135: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_67, torch.float32);  reshape_default_67 = None
        mul_tensor_23: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_67, arg50_1);  arg50_1 = None
        sum_dim_int_list_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_67, [0, 1]);  add_67 = None
        sum_dim_int_list_116: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
        reshape_default_68: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, [768]);  sum_dim_int_list_116 = None
        convert_element_type_default_136: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_default_137: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_68, torch.float32);  reshape_default_68 = None
        sum_dim_int_list_117: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_342, [0], True);  view_342 = None
        reshape_default_69: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, [3072]);  sum_dim_int_list_117 = None
        convert_element_type_default_138: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_default_139: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_69, torch.float32);  reshape_default_69 = None
        mul_tensor_24: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_70, arg45_1);  arg45_1 = None
        sum_dim_int_list_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_70, [0, 1]);  add_70 = None
        sum_dim_int_list_120: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_345, [0], True);  view_345 = None
        reshape_default_70: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, [768]);  sum_dim_int_list_120 = None
        convert_element_type_default_140: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_default_141: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_70, torch.float32);  reshape_default_70 = None
        sum_dim_int_list_121: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_357, [0], True);  view_357 = None
        reshape_default_71: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_121, [768]);  sum_dim_int_list_121 = None
        reshape_default_72: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(mm_142, [1, 512, 768]);  mm_142 = None
        convert_element_type_default_142: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_72, torch.float32);  reshape_default_72 = None
        add_tensor: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_356, convert_element_type_default_142);  mul_356 = convert_element_type_default_142 = None
        convert_element_type_default_143: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_default_144: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_71, torch.float32);  reshape_default_71 = None
        sum_dim_int_list_122: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
        reshape_default_73: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_122, [768]);  sum_dim_int_list_122 = None
        reshape_default_74: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(mm_144, [1, 512, 768]);  mm_144 = None
        convert_element_type_default_145: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_74, torch.float32);  reshape_default_74 = None
        add_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_145);  add_tensor = convert_element_type_default_145 = None
        convert_element_type_default_146: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_default_147: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_73, torch.float32);  reshape_default_73 = None
        sum_dim_int_list_123: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_365, [0], True);  view_365 = None
        reshape_default_75: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_123, [768]);  sum_dim_int_list_123 = None
        reshape_default_76: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(mm_146, [1, 512, 768]);  mm_146 = None
        convert_element_type_default_148: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_76, torch.float32);  reshape_default_76 = None
        add_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_148);  add_tensor_1 = convert_element_type_default_148 = None
        convert_element_type_default_149: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_default_150: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_75, torch.float32);  reshape_default_75 = None
        convert_element_type_default_151: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(arg37_1, torch.float32);  arg37_1 = None
        mul_tensor_25: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_151, 1.1111111111111112);  convert_element_type_default_151 = None
        mul_tensor_26: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_25);  add_tensor_2 = mul_tensor_25 = None
        mul_tensor_27: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_26, arg2_1);  arg2_1 = None
        mul_tensor_28: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 768)
        sum_dim_int_list_124: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [2], True)
        mul_tensor_29: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_27, arg36_1);  mul_tensor_27 = None
        sum_dim_int_list_125: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [2], True);  mul_tensor_29 = None
        mul_tensor_30: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg36_1, sum_dim_int_list_125);  sum_dim_int_list_125 = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_28, sum_dim_int_list_124);  mul_tensor_28 = sum_dim_int_list_124 = None
        sub_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_30);  sub_tensor = mul_tensor_30 = None
        mul_tensor_31: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg349_1, sub_tensor_1);  arg349_1 = sub_tensor_1 = None
        mul_tensor_32: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_26, arg36_1);  arg36_1 = None
        sum_dim_int_list_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None
        full_default: "b8[1, 512, 1]" = torch.ops.aten.full.default([1, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1, 512, 768]" = torch.ops.aten.where.self(full_default, full_1, mul_tensor_31);  full_default = None
        full_default_1: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2, 768]" = torch.ops.aten.index_put.default(full_default_1, [arg29_1], where_self, True);  full_default_1 = arg29_1 = where_self = None
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg35_1, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self_1: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_1, mul_tensor_31);  unsqueeze_default = None
        full_default_2: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_2, [arg35_1], where_self_1, True);  arg35_1 = where_self_1 = None
        eq_scalar_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg34_1, -1)
        unsqueeze_default_1: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_2: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_1, mul_tensor_31);  unsqueeze_default_1 = None
        index_put_default_2: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_2, [arg34_1], where_self_2, True);  arg34_1 = where_self_2 = None
        eq_scalar_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg33_1, -1)
        unsqueeze_default_2: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_3: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_2, full_1, mul_tensor_31);  unsqueeze_default_2 = None
        index_put_default_3: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_2, [arg33_1], where_self_3, True);  arg33_1 = where_self_3 = None
        eq_scalar_3: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg32_1, -1)
        unsqueeze_default_3: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_3, -1);  eq_scalar_3 = None
        where_self_4: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_3, full_1, mul_tensor_31);  unsqueeze_default_3 = None
        index_put_default_4: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_2, [arg32_1], where_self_4, True);  arg32_1 = where_self_4 = None
        eq_scalar_4: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg31_1, -1)
        unsqueeze_default_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_4, -1);  eq_scalar_4 = None
        where_self_5: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_4, full_1, mul_tensor_31);  unsqueeze_default_4 = None
        index_put_default_5: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_2, [arg31_1], where_self_5, True);  arg31_1 = where_self_5 = None
        add_tensor_3: "f32[1024, 768]" = torch.ops.aten.add.Tensor(index_put_default_3, index_put_default_5);  index_put_default_3 = index_put_default_5 = None
        eq_scalar_5: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg30_1, -1)
        unsqueeze_default_5: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_5, -1);  eq_scalar_5 = None
        where_self_6: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_5, full_1, mul_tensor_31);  unsqueeze_default_5 = None
        index_put_default_6: "f32[1024, 768]" = torch.ops.aten.index_put.default(full_default_2, [arg30_1], where_self_6, True);  full_default_2 = arg30_1 = where_self_6 = None
        add_tensor_4: "f32[1024, 768]" = torch.ops.aten.add.Tensor(index_put_default_4, index_put_default_6);  index_put_default_4 = index_put_default_6 = None
        eq_scalar_6: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg1_1, -1)
        unsqueeze_default_6: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_6, -1);  eq_scalar_6 = None
        where_self_7: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_6, full_1, mul_tensor_31);  unsqueeze_default_6 = None
        full_default_3: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_7: "f32[512, 768]" = torch.ops.aten.index_put.default(full_default_3, [arg1_1], where_self_7, True);  full_default_3 = arg1_1 = where_self_7 = None
        eq_scalar_7: "b8[1, 512]" = torch.ops.aten.eq.Scalar(arg0_1, 0)
        unsqueeze_default_7: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_7, -1);  eq_scalar_7 = None
        where_self_8: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default_7, full_1, mul_tensor_31);  unsqueeze_default_7 = full_1 = mul_tensor_31 = None
        full_default_4: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_8: "f32[30522, 768]" = torch.ops.aten.index_put.default(full_default_4, [arg0_1], where_self_8, True);  full_default_4 = arg0_1 = where_self_8 = None
        add_tensor_5: "f32[30522, 768]" = torch.ops.aten.add.Tensor(convert_element_type_default, index_put_default_8);  convert_element_type_default = index_put_default_8 = None
        _output_to_half_0: "f16[30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float16);  convert_element_type_default_1 = None
        _output_to_half_1: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_1, torch.float16);  sum_dim_int_list_1 = None
        _output_to_half_2: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_2, torch.float16);  sum_dim_int_list_2 = None
        _output_to_half_3: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float16);  convert_element_type_default_2 = None
        _output_to_half_4: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float16);  convert_element_type_default_3 = None
        _output_to_half_5: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_4, torch.float16);  sum_dim_int_list_4 = None
        _output_to_half_6: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_5, torch.float16);  sum_dim_int_list_5 = None
        _output_to_half_7: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float16);  convert_element_type_default_4 = None
        _output_to_half_8: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float16);  convert_element_type_default_5 = None
        _output_to_half_9: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_6, torch.float16);  convert_element_type_default_6 = None
        _output_to_half_10: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_7, torch.float16);  convert_element_type_default_7 = None
        _output_to_half_11: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_8, torch.float16);  sum_dim_int_list_8 = None
        _output_to_half_12: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_9, torch.float16);  sum_dim_int_list_9 = None
        _output_to_half_13: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_8, torch.float16);  convert_element_type_default_8 = None
        _output_to_half_14: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_9, torch.float16);  convert_element_type_default_9 = None
        _output_to_half_15: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_10, torch.float16);  convert_element_type_default_10 = None
        _output_to_half_16: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_11, torch.float16);  convert_element_type_default_11 = None
        _output_to_half_17: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_12, torch.float16);  convert_element_type_default_12 = None
        _output_to_half_18: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_13, torch.float16);  convert_element_type_default_13 = None
        _output_to_half_19: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_14, torch.float16);  convert_element_type_default_14 = None
        _output_to_half_20: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_15, torch.float16);  convert_element_type_default_15 = None
        _output_to_half_21: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_14, torch.float16);  sum_dim_int_list_14 = None
        _output_to_half_22: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_15, torch.float16);  sum_dim_int_list_15 = None
        _output_to_half_23: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_16, torch.float16);  convert_element_type_default_16 = None
        _output_to_half_24: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_17, torch.float16);  convert_element_type_default_17 = None
        _output_to_half_25: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_18, torch.float16);  convert_element_type_default_18 = None
        _output_to_half_26: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_19, torch.float16);  convert_element_type_default_19 = None
        _output_to_half_27: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_18, torch.float16);  sum_dim_int_list_18 = None
        _output_to_half_28: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_19, torch.float16);  sum_dim_int_list_19 = None
        _output_to_half_29: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_20, torch.float16);  convert_element_type_default_20 = None
        _output_to_half_30: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_21, torch.float16);  convert_element_type_default_21 = None
        _output_to_half_31: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_22, torch.float16);  convert_element_type_default_22 = None
        _output_to_half_32: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_23, torch.float16);  convert_element_type_default_23 = None
        _output_to_half_33: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_24, torch.float16);  convert_element_type_default_24 = None
        _output_to_half_34: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_25, torch.float16);  convert_element_type_default_25 = None
        _output_to_half_35: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_26, torch.float16);  convert_element_type_default_26 = None
        _output_to_half_36: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_27, torch.float16);  convert_element_type_default_27 = None
        _output_to_half_37: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_24, torch.float16);  sum_dim_int_list_24 = None
        _output_to_half_38: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_25, torch.float16);  sum_dim_int_list_25 = None
        _output_to_half_39: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_28, torch.float16);  convert_element_type_default_28 = None
        _output_to_half_40: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_29, torch.float16);  convert_element_type_default_29 = None
        _output_to_half_41: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_30, torch.float16);  convert_element_type_default_30 = None
        _output_to_half_42: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_31, torch.float16);  convert_element_type_default_31 = None
        _output_to_half_43: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_28, torch.float16);  sum_dim_int_list_28 = None
        _output_to_half_44: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_29, torch.float16);  sum_dim_int_list_29 = None
        _output_to_half_45: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_32, torch.float16);  convert_element_type_default_32 = None
        _output_to_half_46: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_33, torch.float16);  convert_element_type_default_33 = None
        _output_to_half_47: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_34, torch.float16);  convert_element_type_default_34 = None
        _output_to_half_48: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_35, torch.float16);  convert_element_type_default_35 = None
        _output_to_half_49: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_36, torch.float16);  convert_element_type_default_36 = None
        _output_to_half_50: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_37, torch.float16);  convert_element_type_default_37 = None
        _output_to_half_51: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_38, torch.float16);  convert_element_type_default_38 = None
        _output_to_half_52: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_39, torch.float16);  convert_element_type_default_39 = None
        _output_to_half_53: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_34, torch.float16);  sum_dim_int_list_34 = None
        _output_to_half_54: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_35, torch.float16);  sum_dim_int_list_35 = None
        _output_to_half_55: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_40, torch.float16);  convert_element_type_default_40 = None
        _output_to_half_56: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_41, torch.float16);  convert_element_type_default_41 = None
        _output_to_half_57: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_42, torch.float16);  convert_element_type_default_42 = None
        _output_to_half_58: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_43, torch.float16);  convert_element_type_default_43 = None
        _output_to_half_59: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_38, torch.float16);  sum_dim_int_list_38 = None
        _output_to_half_60: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_39, torch.float16);  sum_dim_int_list_39 = None
        _output_to_half_61: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_44, torch.float16);  convert_element_type_default_44 = None
        _output_to_half_62: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_45, torch.float16);  convert_element_type_default_45 = None
        _output_to_half_63: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_46, torch.float16);  convert_element_type_default_46 = None
        _output_to_half_64: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_47, torch.float16);  convert_element_type_default_47 = None
        _output_to_half_65: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_48, torch.float16);  convert_element_type_default_48 = None
        _output_to_half_66: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_49, torch.float16);  convert_element_type_default_49 = None
        _output_to_half_67: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_50, torch.float16);  convert_element_type_default_50 = None
        _output_to_half_68: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_51, torch.float16);  convert_element_type_default_51 = None
        _output_to_half_69: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_44, torch.float16);  sum_dim_int_list_44 = None
        _output_to_half_70: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_45, torch.float16);  sum_dim_int_list_45 = None
        _output_to_half_71: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_52, torch.float16);  convert_element_type_default_52 = None
        _output_to_half_72: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_53, torch.float16);  convert_element_type_default_53 = None
        _output_to_half_73: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_54, torch.float16);  convert_element_type_default_54 = None
        _output_to_half_74: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_55, torch.float16);  convert_element_type_default_55 = None
        _output_to_half_75: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_48, torch.float16);  sum_dim_int_list_48 = None
        _output_to_half_76: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_49, torch.float16);  sum_dim_int_list_49 = None
        _output_to_half_77: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_56, torch.float16);  convert_element_type_default_56 = None
        _output_to_half_78: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_57, torch.float16);  convert_element_type_default_57 = None
        _output_to_half_79: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_58, torch.float16);  convert_element_type_default_58 = None
        _output_to_half_80: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_59, torch.float16);  convert_element_type_default_59 = None
        _output_to_half_81: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_60, torch.float16);  convert_element_type_default_60 = None
        _output_to_half_82: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_61, torch.float16);  convert_element_type_default_61 = None
        _output_to_half_83: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_62, torch.float16);  convert_element_type_default_62 = None
        _output_to_half_84: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_63, torch.float16);  convert_element_type_default_63 = None
        _output_to_half_85: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_54, torch.float16);  sum_dim_int_list_54 = None
        _output_to_half_86: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_55, torch.float16);  sum_dim_int_list_55 = None
        _output_to_half_87: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_64, torch.float16);  convert_element_type_default_64 = None
        _output_to_half_88: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_65, torch.float16);  convert_element_type_default_65 = None
        _output_to_half_89: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_66, torch.float16);  convert_element_type_default_66 = None
        _output_to_half_90: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_67, torch.float16);  convert_element_type_default_67 = None
        _output_to_half_91: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_58, torch.float16);  sum_dim_int_list_58 = None
        _output_to_half_92: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_59, torch.float16);  sum_dim_int_list_59 = None
        _output_to_half_93: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_68, torch.float16);  convert_element_type_default_68 = None
        _output_to_half_94: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_69, torch.float16);  convert_element_type_default_69 = None
        _output_to_half_95: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_70, torch.float16);  convert_element_type_default_70 = None
        _output_to_half_96: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_71, torch.float16);  convert_element_type_default_71 = None
        _output_to_half_97: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_72, torch.float16);  convert_element_type_default_72 = None
        _output_to_half_98: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_73, torch.float16);  convert_element_type_default_73 = None
        _output_to_half_99: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_74, torch.float16);  convert_element_type_default_74 = None
        _output_to_half_100: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_75, torch.float16);  convert_element_type_default_75 = None
        _output_to_half_101: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_64, torch.float16);  sum_dim_int_list_64 = None
        _output_to_half_102: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_65, torch.float16);  sum_dim_int_list_65 = None
        _output_to_half_103: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_76, torch.float16);  convert_element_type_default_76 = None
        _output_to_half_104: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_77, torch.float16);  convert_element_type_default_77 = None
        _output_to_half_105: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_78, torch.float16);  convert_element_type_default_78 = None
        _output_to_half_106: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_79, torch.float16);  convert_element_type_default_79 = None
        _output_to_half_107: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_68, torch.float16);  sum_dim_int_list_68 = None
        _output_to_half_108: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_69, torch.float16);  sum_dim_int_list_69 = None
        _output_to_half_109: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_80, torch.float16);  convert_element_type_default_80 = None
        _output_to_half_110: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_81, torch.float16);  convert_element_type_default_81 = None
        _output_to_half_111: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_82, torch.float16);  convert_element_type_default_82 = None
        _output_to_half_112: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_83, torch.float16);  convert_element_type_default_83 = None
        _output_to_half_113: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_84, torch.float16);  convert_element_type_default_84 = None
        _output_to_half_114: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_85, torch.float16);  convert_element_type_default_85 = None
        _output_to_half_115: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_86, torch.float16);  convert_element_type_default_86 = None
        _output_to_half_116: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_87, torch.float16);  convert_element_type_default_87 = None
        _output_to_half_117: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_74, torch.float16);  sum_dim_int_list_74 = None
        _output_to_half_118: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_75, torch.float16);  sum_dim_int_list_75 = None
        _output_to_half_119: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_88, torch.float16);  convert_element_type_default_88 = None
        _output_to_half_120: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_89, torch.float16);  convert_element_type_default_89 = None
        _output_to_half_121: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_90, torch.float16);  convert_element_type_default_90 = None
        _output_to_half_122: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_91, torch.float16);  convert_element_type_default_91 = None
        _output_to_half_123: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_78, torch.float16);  sum_dim_int_list_78 = None
        _output_to_half_124: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_79, torch.float16);  sum_dim_int_list_79 = None
        _output_to_half_125: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_92, torch.float16);  convert_element_type_default_92 = None
        _output_to_half_126: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_93, torch.float16);  convert_element_type_default_93 = None
        _output_to_half_127: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_94, torch.float16);  convert_element_type_default_94 = None
        _output_to_half_128: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_95, torch.float16);  convert_element_type_default_95 = None
        _output_to_half_129: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_96, torch.float16);  convert_element_type_default_96 = None
        _output_to_half_130: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_97, torch.float16);  convert_element_type_default_97 = None
        _output_to_half_131: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_98, torch.float16);  convert_element_type_default_98 = None
        _output_to_half_132: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_99, torch.float16);  convert_element_type_default_99 = None
        _output_to_half_133: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_84, torch.float16);  sum_dim_int_list_84 = None
        _output_to_half_134: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_85, torch.float16);  sum_dim_int_list_85 = None
        _output_to_half_135: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_100, torch.float16);  convert_element_type_default_100 = None
        _output_to_half_136: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_101, torch.float16);  convert_element_type_default_101 = None
        _output_to_half_137: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_102, torch.float16);  convert_element_type_default_102 = None
        _output_to_half_138: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_103, torch.float16);  convert_element_type_default_103 = None
        _output_to_half_139: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_88, torch.float16);  sum_dim_int_list_88 = None
        _output_to_half_140: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_89, torch.float16);  sum_dim_int_list_89 = None
        _output_to_half_141: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_104, torch.float16);  convert_element_type_default_104 = None
        _output_to_half_142: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_105, torch.float16);  convert_element_type_default_105 = None
        _output_to_half_143: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_106, torch.float16);  convert_element_type_default_106 = None
        _output_to_half_144: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_107, torch.float16);  convert_element_type_default_107 = None
        _output_to_half_145: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_108, torch.float16);  convert_element_type_default_108 = None
        _output_to_half_146: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_109, torch.float16);  convert_element_type_default_109 = None
        _output_to_half_147: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_110, torch.float16);  convert_element_type_default_110 = None
        _output_to_half_148: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_111, torch.float16);  convert_element_type_default_111 = None
        _output_to_half_149: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_94, torch.float16);  sum_dim_int_list_94 = None
        _output_to_half_150: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_95, torch.float16);  sum_dim_int_list_95 = None
        _output_to_half_151: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_112, torch.float16);  convert_element_type_default_112 = None
        _output_to_half_152: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_113, torch.float16);  convert_element_type_default_113 = None
        _output_to_half_153: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_114, torch.float16);  convert_element_type_default_114 = None
        _output_to_half_154: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_115, torch.float16);  convert_element_type_default_115 = None
        _output_to_half_155: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_98, torch.float16);  sum_dim_int_list_98 = None
        _output_to_half_156: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_99, torch.float16);  sum_dim_int_list_99 = None
        _output_to_half_157: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_116, torch.float16);  convert_element_type_default_116 = None
        _output_to_half_158: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_117, torch.float16);  convert_element_type_default_117 = None
        _output_to_half_159: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_118, torch.float16);  convert_element_type_default_118 = None
        _output_to_half_160: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_119, torch.float16);  convert_element_type_default_119 = None
        _output_to_half_161: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_120, torch.float16);  convert_element_type_default_120 = None
        _output_to_half_162: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_121, torch.float16);  convert_element_type_default_121 = None
        _output_to_half_163: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_122, torch.float16);  convert_element_type_default_122 = None
        _output_to_half_164: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_123, torch.float16);  convert_element_type_default_123 = None
        _output_to_half_165: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_104, torch.float16);  sum_dim_int_list_104 = None
        _output_to_half_166: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_105, torch.float16);  sum_dim_int_list_105 = None
        _output_to_half_167: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_124, torch.float16);  convert_element_type_default_124 = None
        _output_to_half_168: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_125, torch.float16);  convert_element_type_default_125 = None
        _output_to_half_169: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_126, torch.float16);  convert_element_type_default_126 = None
        _output_to_half_170: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_127, torch.float16);  convert_element_type_default_127 = None
        _output_to_half_171: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_108, torch.float16);  sum_dim_int_list_108 = None
        _output_to_half_172: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_109, torch.float16);  sum_dim_int_list_109 = None
        _output_to_half_173: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_128, torch.float16);  convert_element_type_default_128 = None
        _output_to_half_174: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_129, torch.float16);  convert_element_type_default_129 = None
        _output_to_half_175: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_130, torch.float16);  convert_element_type_default_130 = None
        _output_to_half_176: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_131, torch.float16);  convert_element_type_default_131 = None
        _output_to_half_177: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_132, torch.float16);  convert_element_type_default_132 = None
        _output_to_half_178: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_133, torch.float16);  convert_element_type_default_133 = None
        _output_to_half_179: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_134, torch.float16);  convert_element_type_default_134 = None
        _output_to_half_180: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_135, torch.float16);  convert_element_type_default_135 = None
        _output_to_half_181: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_114, torch.float16);  sum_dim_int_list_114 = None
        _output_to_half_182: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_115, torch.float16);  sum_dim_int_list_115 = None
        _output_to_half_183: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_136, torch.float16);  convert_element_type_default_136 = None
        _output_to_half_184: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_137, torch.float16);  convert_element_type_default_137 = None
        _output_to_half_185: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_138, torch.float16);  convert_element_type_default_138 = None
        _output_to_half_186: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_139, torch.float16);  convert_element_type_default_139 = None
        _output_to_half_187: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_118, torch.float16);  sum_dim_int_list_118 = None
        _output_to_half_188: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_119, torch.float16);  sum_dim_int_list_119 = None
        _output_to_half_189: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_140, torch.float16);  convert_element_type_default_140 = None
        _output_to_half_190: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_141, torch.float16);  convert_element_type_default_141 = None
        _output_to_half_191: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_143, torch.float16);  convert_element_type_default_143 = None
        _output_to_half_192: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_144, torch.float16);  convert_element_type_default_144 = None
        _output_to_half_193: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_146, torch.float16);  convert_element_type_default_146 = None
        _output_to_half_194: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_147, torch.float16);  convert_element_type_default_147 = None
        _output_to_half_195: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_149, torch.float16);  convert_element_type_default_149 = None
        _output_to_half_196: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_150, torch.float16);  convert_element_type_default_150 = None
        _output_to_half_197: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_126, torch.float16);  sum_dim_int_list_126 = None
        _output_to_half_198: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_127, torch.float16);  sum_dim_int_list_127 = None
        _output_to_half_199: "f16[2, 768]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.float16);  index_put_default = None
        _output_to_half_200: "f16[1024, 768]" = torch.ops.prims.convert_element_type.default(index_put_default_1, torch.float16);  index_put_default_1 = None
        _output_to_half_201: "f16[1024, 768]" = torch.ops.prims.convert_element_type.default(index_put_default_2, torch.float16);  index_put_default_2 = None
        _output_to_half_202: "f16[1024, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float16);  add_tensor_3 = None
        _output_to_half_203: "f16[1024, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float16);  add_tensor_4 = None
        _output_to_half_204: "f16[512, 768]" = torch.ops.prims.convert_element_type.default(index_put_default_7, torch.float16);  index_put_default_7 = None
        _output_to_half_205: "f16[30522, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float16);  add_tensor_5 = None
        return (_output_to_half_0, _output_to_half_1, _output_to_half_2, _output_to_half_3, _output_to_half_4, _output_to_half_5, _output_to_half_6, _output_to_half_7, _output_to_half_8, _output_to_half_9, _output_to_half_10, _output_to_half_11, _output_to_half_12, _output_to_half_13, _output_to_half_14, _output_to_half_15, _output_to_half_16, _output_to_half_17, _output_to_half_18, _output_to_half_19, _output_to_half_20, _output_to_half_21, _output_to_half_22, _output_to_half_23, _output_to_half_24, _output_to_half_25, _output_to_half_26, _output_to_half_27, _output_to_half_28, _output_to_half_29, _output_to_half_30, _output_to_half_31, _output_to_half_32, _output_to_half_33, _output_to_half_34, _output_to_half_35, _output_to_half_36, _output_to_half_37, _output_to_half_38, _output_to_half_39, _output_to_half_40, _output_to_half_41, _output_to_half_42, _output_to_half_43, _output_to_half_44, _output_to_half_45, _output_to_half_46, _output_to_half_47, _output_to_half_48, _output_to_half_49, _output_to_half_50, _output_to_half_51, _output_to_half_52, _output_to_half_53, _output_to_half_54, _output_to_half_55, _output_to_half_56, _output_to_half_57, _output_to_half_58, _output_to_half_59, _output_to_half_60, _output_to_half_61, _output_to_half_62, _output_to_half_63, _output_to_half_64, _output_to_half_65, _output_to_half_66, _output_to_half_67, _output_to_half_68, _output_to_half_69, _output_to_half_70, _output_to_half_71, _output_to_half_72, _output_to_half_73, _output_to_half_74, _output_to_half_75, _output_to_half_76, _output_to_half_77, _output_to_half_78, _output_to_half_79, _output_to_half_80, _output_to_half_81, _output_to_half_82, _output_to_half_83, _output_to_half_84, _output_to_half_85, _output_to_half_86, _output_to_half_87, _output_to_half_88, _output_to_half_89, _output_to_half_90, _output_to_half_91, _output_to_half_92, _output_to_half_93, _output_to_half_94, _output_to_half_95, _output_to_half_96, _output_to_half_97, _output_to_half_98, _output_to_half_99, _output_to_half_100, _output_to_half_101, _output_to_half_102, _output_to_half_103, _output_to_half_104, _output_to_half_105, _output_to_half_106, _output_to_half_107, _output_to_half_108, _output_to_half_109, _output_to_half_110, _output_to_half_111, _output_to_half_112, _output_to_half_113, _output_to_half_114, _output_to_half_115, _output_to_half_116, _output_to_half_117, _output_to_half_118, _output_to_half_119, _output_to_half_120, _output_to_half_121, _output_to_half_122, _output_to_half_123, _output_to_half_124, _output_to_half_125, _output_to_half_126, _output_to_half_127, _output_to_half_128, _output_to_half_129, _output_to_half_130, _output_to_half_131, _output_to_half_132, _output_to_half_133, _output_to_half_134, _output_to_half_135, _output_to_half_136, _output_to_half_137, _output_to_half_138, _output_to_half_139, _output_to_half_140, _output_to_half_141, _output_to_half_142, _output_to_half_143, _output_to_half_144, _output_to_half_145, _output_to_half_146, _output_to_half_147, _output_to_half_148, _output_to_half_149, _output_to_half_150, _output_to_half_151, _output_to_half_152, _output_to_half_153, _output_to_half_154, _output_to_half_155, _output_to_half_156, _output_to_half_157, _output_to_half_158, _output_to_half_159, _output_to_half_160, _output_to_half_161, _output_to_half_162, _output_to_half_163, _output_to_half_164, _output_to_half_165, _output_to_half_166, _output_to_half_167, _output_to_half_168, _output_to_half_169, _output_to_half_170, _output_to_half_171, _output_to_half_172, _output_to_half_173, _output_to_half_174, _output_to_half_175, _output_to_half_176, _output_to_half_177, _output_to_half_178, _output_to_half_179, _output_to_half_180, _output_to_half_181, _output_to_half_182, _output_to_half_183, _output_to_half_184, _output_to_half_185, _output_to_half_186, _output_to_half_187, _output_to_half_188, _output_to_half_189, _output_to_half_190, _output_to_half_191, _output_to_half_192, _output_to_half_193, _output_to_half_194, _output_to_half_195, _output_to_half_196, _output_to_half_197, _output_to_half_198, _output_to_half_199, _output_to_half_200, _output_to_half_201, _output_to_half_202, _output_to_half_203, _output_to_half_204, _output_to_half_205)


def _default_make_inputs():
    return [
    torch.randn([512, 30522], dtype=torch.float16, device='cuda'),
    torch.randn([30522, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_31
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_61
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_91
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_121
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_151
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_181
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_211
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_241
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_271
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_301
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_331
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([512, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn(393216, dtype=torch.float16, device='cuda').as_strided([512, 768], [1, 512]),  # view_361
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [1, 512, 768], dtype=torch.bool, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, (2045,), dtype=torch.int64, device='cuda').as_strided([1, 512], [2048, 4]),  # arg33_1
    torch.randint(0, 2, (2045,), dtype=torch.int64, device='cuda').as_strided([1, 512], [2048, 4]),  # arg32_1
    torch.randint(0, 2, (2045,), dtype=torch.int64, device='cuda').as_strided([1, 512], [2048, 4]),  # arg31_1
    torch.randint(0, 2, (2045,), dtype=torch.int64, device='cuda').as_strided([1, 512], [2048, 4]),  # arg30_1
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 512], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
