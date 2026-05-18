"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: efb5551a7c94
Shape hash: f9e9c662
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
    def forward(self, mm: "f16[512, 512]", index_put: "f32[32, 77, 512]", arg353_1: "f32[32, 77, 512]", view: "f16[2464, 512]", mm_3: "f16[512, 2048]", view_4: "f16[2464, 2048]", mm_5: "f16[2048, 512]", convert_element_type_12: "f32[77, 32, 512]", arg364_1: "f32[77, 32, 512]", view_7: "f16[2464, 512]", mm_7: "f16[512, 512]", view_18: "f16[2464, 1536]", mm_9: "f16[1536, 512]", convert_element_type_18: "f32[77, 32, 512]", arg368_1: "f32[77, 32, 512]", view_21: "f16[2464, 512]", mm_11: "f16[512, 2048]", view_25: "f16[2464, 2048]", mm_13: "f16[2048, 512]", convert_element_type_27: "f32[77, 32, 512]", arg372_1: "f32[77, 32, 512]", view_28: "f16[2464, 512]", mm_15: "f16[512, 512]", view_39: "f16[2464, 1536]", mm_17: "f16[1536, 512]", convert_element_type_33: "f32[77, 32, 512]", arg376_1: "f32[77, 32, 512]", view_42: "f16[2464, 512]", mm_19: "f16[512, 2048]", view_46: "f16[2464, 2048]", mm_21: "f16[2048, 512]", convert_element_type_42: "f32[77, 32, 512]", arg380_1: "f32[77, 32, 512]", view_49: "f16[2464, 512]", mm_23: "f16[512, 512]", view_60: "f16[2464, 1536]", mm_25: "f16[1536, 512]", convert_element_type_48: "f32[77, 32, 512]", arg384_1: "f32[77, 32, 512]", view_63: "f16[2464, 512]", mm_27: "f16[512, 2048]", view_67: "f16[2464, 2048]", mm_29: "f16[2048, 512]", convert_element_type_57: "f32[77, 32, 512]", arg388_1: "f32[77, 32, 512]", view_70: "f16[2464, 512]", mm_31: "f16[512, 512]", view_81: "f16[2464, 1536]", mm_33: "f16[1536, 512]", convert_element_type_63: "f32[77, 32, 512]", arg392_1: "f32[77, 32, 512]", view_84: "f16[2464, 512]", mm_35: "f16[512, 2048]", view_88: "f16[2464, 2048]", mm_37: "f16[2048, 512]", convert_element_type_72: "f32[77, 32, 512]", arg396_1: "f32[77, 32, 512]", view_91: "f16[2464, 512]", mm_39: "f16[512, 512]", view_102: "f16[2464, 1536]", mm_41: "f16[1536, 512]", convert_element_type_78: "f32[77, 32, 512]", arg400_1: "f32[77, 32, 512]", view_105: "f16[2464, 512]", mm_43: "f16[512, 2048]", view_109: "f16[2464, 2048]", mm_45: "f16[2048, 512]", convert_element_type_87: "f32[77, 32, 512]", arg404_1: "f32[77, 32, 512]", view_112: "f16[2464, 512]", mm_47: "f16[512, 512]", view_123: "f16[2464, 1536]", mm_49: "f16[1536, 512]", convert_element_type_93: "f32[77, 32, 512]", arg408_1: "f32[77, 32, 512]", view_126: "f16[2464, 512]", mm_51: "f16[512, 2048]", view_130: "f16[2464, 2048]", mm_53: "f16[2048, 512]", convert_element_type_102: "f32[77, 32, 512]", arg412_1: "f32[77, 32, 512]", view_133: "f16[2464, 512]", mm_55: "f16[512, 512]", view_144: "f16[2464, 1536]", mm_57: "f16[1536, 512]", convert_element_type_108: "f32[77, 32, 512]", arg416_1: "f32[77, 32, 512]", view_147: "f16[2464, 512]", mm_59: "f16[512, 2048]", view_151: "f16[2464, 2048]", mm_61: "f16[2048, 512]", convert_element_type_117: "f32[77, 32, 512]", arg420_1: "f32[77, 32, 512]", view_154: "f16[2464, 512]", mm_63: "f16[512, 512]", view_165: "f16[2464, 1536]", mm_65: "f16[1536, 512]", convert_element_type_123: "f32[77, 32, 512]", arg424_1: "f32[77, 32, 512]", view_168: "f16[2464, 512]", mm_67: "f16[512, 2048]", view_172: "f16[2464, 2048]", mm_69: "f16[2048, 512]", convert_element_type_132: "f32[77, 32, 512]", arg428_1: "f32[77, 32, 512]", view_175: "f16[2464, 512]", mm_71: "f16[512, 512]", view_186: "f16[2464, 1536]", mm_73: "f16[1536, 512]", convert_element_type_138: "f32[77, 32, 512]", arg432_1: "f32[77, 32, 512]", view_189: "f16[2464, 512]", mm_75: "f16[512, 2048]", view_193: "f16[2464, 2048]", mm_77: "f16[2048, 512]", convert_element_type_147: "f32[77, 32, 512]", arg436_1: "f32[77, 32, 512]", view_196: "f16[2464, 512]", mm_79: "f16[512, 512]", view_207: "f16[2464, 1536]", mm_81: "f16[1536, 512]", convert_element_type_153: "f32[77, 32, 512]", arg440_1: "f32[77, 32, 512]", view_210: "f16[2464, 512]", mm_83: "f16[512, 2048]", view_214: "f16[2464, 2048]", mm_85: "f16[2048, 512]", convert_element_type_162: "f32[77, 32, 512]", arg444_1: "f32[77, 32, 512]", view_217: "f16[2464, 512]", mm_87: "f16[512, 512]", view_228: "f16[2464, 1536]", mm_89: "f16[1536, 512]", convert_element_type_168: "f32[77, 32, 512]", arg448_1: "f32[77, 32, 512]", view_231: "f16[2464, 512]", mm_91: "f16[512, 2048]", view_235: "f16[2464, 2048]", mm_93: "f16[2048, 512]", convert_element_type_177: "f32[77, 32, 512]", arg452_1: "f32[77, 32, 512]", view_238: "f16[2464, 512]", mm_95: "f16[512, 512]", view_249: "f16[2464, 1536]", mm_96: "f16[2464, 512]", mm_97: "f16[1536, 512]", arg218_1: "f32[32, 77, 512]", arg29_1: "f32[77, 512]", arg219_1: "f32[77, 32, 1]", arg220_1: "f32[77, 32, 1]", arg30_1: "f32[512]", add_58: "f32[77, 32, 512]", arg28_1: "i64[32, 77]", full: "f32[]", mm_98: "f16[768, 512]", convert_element_type_186: "f32[32, 768]", arg458_1: "f32[32, 768]", view_253: "f16[1600, 768]", mm_101: "f16[768, 3072]", view_257: "f16[1600, 3072]", mm_103: "f16[3072, 768]", convert_element_type_194: "f32[32, 50, 768]", arg213_1: "f32[32, 50, 768]", view_260: "f16[1600, 768]", mm_105: "f16[768, 768]", view_270: "f16[50, 32, 2304]", mm_106: "f16[2304, 768]", permute_135: "f32[32, 50, 768]", arg204_1: "f32[32, 50, 768]", view_274: "f16[1600, 768]", mm_109: "f16[768, 3072]", view_278: "f16[1600, 3072]", mm_111: "f16[3072, 768]", convert_element_type_209: "f32[32, 50, 768]", arg200_1: "f32[32, 50, 768]", view_281: "f16[1600, 768]", mm_113: "f16[768, 768]", view_291: "f16[50, 32, 2304]", mm_114: "f16[2304, 768]", permute_147: "f32[32, 50, 768]", arg191_1: "f32[32, 50, 768]", view_295: "f16[1600, 768]", mm_117: "f16[768, 3072]", view_299: "f16[1600, 3072]", mm_119: "f16[3072, 768]", convert_element_type_224: "f32[32, 50, 768]", arg187_1: "f32[32, 50, 768]", view_302: "f16[1600, 768]", mm_121: "f16[768, 768]", view_312: "f16[50, 32, 2304]", mm_122: "f16[2304, 768]", permute_159: "f32[32, 50, 768]", arg178_1: "f32[32, 50, 768]", view_316: "f16[1600, 768]", mm_125: "f16[768, 3072]", view_320: "f16[1600, 3072]", mm_127: "f16[3072, 768]", convert_element_type_239: "f32[32, 50, 768]", arg174_1: "f32[32, 50, 768]", view_323: "f16[1600, 768]", mm_129: "f16[768, 768]", view_333: "f16[50, 32, 2304]", mm_130: "f16[2304, 768]", permute_171: "f32[32, 50, 768]", arg165_1: "f32[32, 50, 768]", view_337: "f16[1600, 768]", mm_133: "f16[768, 3072]", view_341: "f16[1600, 3072]", mm_135: "f16[3072, 768]", convert_element_type_254: "f32[32, 50, 768]", arg161_1: "f32[32, 50, 768]", view_344: "f16[1600, 768]", mm_137: "f16[768, 768]", view_354: "f16[50, 32, 2304]", mm_138: "f16[2304, 768]", permute_183: "f32[32, 50, 768]", arg152_1: "f32[32, 50, 768]", view_358: "f16[1600, 768]", mm_141: "f16[768, 3072]", view_362: "f16[1600, 3072]", mm_143: "f16[3072, 768]", convert_element_type_269: "f32[32, 50, 768]", arg148_1: "f32[32, 50, 768]", view_365: "f16[1600, 768]", mm_145: "f16[768, 768]", view_375: "f16[50, 32, 2304]", mm_146: "f16[2304, 768]", permute_195: "f32[32, 50, 768]", arg139_1: "f32[32, 50, 768]", view_379: "f16[1600, 768]", mm_149: "f16[768, 3072]", view_383: "f16[1600, 3072]", mm_151: "f16[3072, 768]", convert_element_type_284: "f32[32, 50, 768]", arg135_1: "f32[32, 50, 768]", view_386: "f16[1600, 768]", mm_153: "f16[768, 768]", view_396: "f16[50, 32, 2304]", mm_154: "f16[2304, 768]", permute_207: "f32[32, 50, 768]", arg126_1: "f32[32, 50, 768]", view_400: "f16[1600, 768]", mm_157: "f16[768, 3072]", view_404: "f16[1600, 3072]", mm_159: "f16[3072, 768]", convert_element_type_299: "f32[32, 50, 768]", arg122_1: "f32[32, 50, 768]", view_407: "f16[1600, 768]", mm_161: "f16[768, 768]", view_417: "f16[50, 32, 2304]", mm_162: "f16[2304, 768]", permute_219: "f32[32, 50, 768]", arg113_1: "f32[32, 50, 768]", view_421: "f16[1600, 768]", mm_165: "f16[768, 3072]", view_425: "f16[1600, 3072]", mm_167: "f16[3072, 768]", convert_element_type_314: "f32[32, 50, 768]", arg109_1: "f32[32, 50, 768]", view_428: "f16[1600, 768]", mm_169: "f16[768, 768]", view_438: "f16[50, 32, 2304]", mm_170: "f16[2304, 768]", permute_231: "f32[32, 50, 768]", arg100_1: "f32[32, 50, 768]", view_442: "f16[1600, 768]", mm_173: "f16[768, 3072]", view_446: "f16[1600, 3072]", mm_175: "f16[3072, 768]", convert_element_type_329: "f32[32, 50, 768]", arg96_1: "f32[32, 50, 768]", view_449: "f16[1600, 768]", mm_177: "f16[768, 768]", view_459: "f16[50, 32, 2304]", mm_178: "f16[2304, 768]", permute_243: "f32[32, 50, 768]", arg87_1: "f32[32, 50, 768]", view_463: "f16[1600, 768]", mm_181: "f16[768, 3072]", view_467: "f16[1600, 3072]", mm_183: "f16[3072, 768]", convert_element_type_344: "f32[32, 50, 768]", arg83_1: "f32[32, 50, 768]", view_470: "f16[1600, 768]", mm_185: "f16[768, 768]", view_480: "f16[50, 32, 2304]", mm_186: "f16[2304, 768]", permute_255: "f32[32, 50, 768]", arg74_1: "f32[32, 50, 768]", view_484: "f16[1600, 768]", mm_189: "f16[768, 3072]", view_488: "f16[1600, 3072]", mm_191: "f16[3072, 768]", convert_element_type_359: "f32[32, 50, 768]", arg70_1: "f32[32, 50, 768]", view_491: "f16[1600, 768]", mm_193: "f16[768, 768]", view_501: "f16[50, 32, 2304]", mm_194: "f16[2304, 768]", permute_267: "f32[32, 50, 768]", mul_447: "f32[32, 50, 768]", add_124: "f32[32, 50, 768]", mul_445: "f32[32, 50, 768]", mul_456: "f32[32, 50, 768]", getitem_73: "f16[768, 3, 32, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mul_tensor: "f32[32, 77, 512]" = torch.ops.aten.mul.Tensor(index_put, arg353_1);  arg353_1 = None
        sum_dim_int_list: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_1: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_put, [0, 1]);  index_put = None
        sum_dim_int_list_2: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view, [0], True);  view = None
        reshape_default: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_0);  sum_dim_int_list_2 = _shape_param_0 = None
        convert_element_type_default_1: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_default_2: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        sum_dim_int_list_3: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True);  view_4 = None
        reshape_default_1: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None
        convert_element_type_default_3: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_default_4: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        mul_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_12, arg364_1);  arg364_1 = None
        sum_dim_int_list_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_12, [0, 1]);  convert_element_type_12 = None
        sum_dim_int_list_6: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True);  view_7 = None
        reshape_default_2: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None
        convert_element_type_default_5: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_default_6: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_2, torch.float32);  reshape_default_2 = None
        sum_dim_int_list_7: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_18, [0], True);  view_18 = None
        reshape_default_3: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None
        convert_element_type_default_7: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_default_8: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_3, torch.float32);  reshape_default_3 = None
        mul_tensor_2: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_18, arg368_1);  arg368_1 = None
        sum_dim_int_list_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_18, [0, 1]);  convert_element_type_18 = None
        sum_dim_int_list_10: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_21, [0], True);  view_21 = None
        reshape_default_4: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_4);  sum_dim_int_list_10 = _shape_param_4 = None
        convert_element_type_default_9: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_default_10: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_4, torch.float32);  reshape_default_4 = None
        sum_dim_int_list_11: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_25, [0], True);  view_25 = None
        reshape_default_5: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None
        convert_element_type_default_11: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_default_12: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_5, torch.float32);  reshape_default_5 = None
        mul_tensor_3: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, arg372_1);  arg372_1 = None
        sum_dim_int_list_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_27, [0, 1]);  convert_element_type_27 = None
        sum_dim_int_list_14: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True);  view_28 = None
        reshape_default_6: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_6);  sum_dim_int_list_14 = _shape_param_6 = None
        convert_element_type_default_13: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_default_14: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_6, torch.float32);  reshape_default_6 = None
        sum_dim_int_list_15: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_39, [0], True);  view_39 = None
        reshape_default_7: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_7);  sum_dim_int_list_15 = _shape_param_7 = None
        convert_element_type_default_15: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_default_16: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_7, torch.float32);  reshape_default_7 = None
        mul_tensor_4: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, arg376_1);  arg376_1 = None
        sum_dim_int_list_16: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_17: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_33, [0, 1]);  convert_element_type_33 = None
        sum_dim_int_list_18: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_42, [0], True);  view_42 = None
        reshape_default_8: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_8);  sum_dim_int_list_18 = _shape_param_8 = None
        convert_element_type_default_17: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_default_18: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_8, torch.float32);  reshape_default_8 = None
        sum_dim_int_list_19: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_46, [0], True);  view_46 = None
        reshape_default_9: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_9);  sum_dim_int_list_19 = _shape_param_9 = None
        convert_element_type_default_19: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_default_20: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_9, torch.float32);  reshape_default_9 = None
        mul_tensor_5: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_42, arg380_1);  arg380_1 = None
        sum_dim_int_list_20: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_21: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_42, [0, 1]);  convert_element_type_42 = None
        sum_dim_int_list_22: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_49, [0], True);  view_49 = None
        reshape_default_10: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_10);  sum_dim_int_list_22 = _shape_param_10 = None
        convert_element_type_default_21: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_default_22: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_10, torch.float32);  reshape_default_10 = None
        sum_dim_int_list_23: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_60, [0], True);  view_60 = None
        reshape_default_11: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_11);  sum_dim_int_list_23 = _shape_param_11 = None
        convert_element_type_default_23: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_default_24: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_11, torch.float32);  reshape_default_11 = None
        mul_tensor_6: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_48, arg384_1);  arg384_1 = None
        sum_dim_int_list_24: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_48, [0, 1]);  convert_element_type_48 = None
        sum_dim_int_list_26: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_63, [0], True);  view_63 = None
        reshape_default_12: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_12);  sum_dim_int_list_26 = _shape_param_12 = None
        convert_element_type_default_25: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_default_26: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_12, torch.float32);  reshape_default_12 = None
        sum_dim_int_list_27: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_67, [0], True);  view_67 = None
        reshape_default_13: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_13);  sum_dim_int_list_27 = _shape_param_13 = None
        convert_element_type_default_27: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_default_28: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_13, torch.float32);  reshape_default_13 = None
        mul_tensor_7: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_57, arg388_1);  arg388_1 = None
        sum_dim_int_list_28: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_29: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_57, [0, 1]);  convert_element_type_57 = None
        sum_dim_int_list_30: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_70, [0], True);  view_70 = None
        reshape_default_14: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_14);  sum_dim_int_list_30 = _shape_param_14 = None
        convert_element_type_default_29: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_default_30: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_14, torch.float32);  reshape_default_14 = None
        sum_dim_int_list_31: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_81, [0], True);  view_81 = None
        reshape_default_15: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_15);  sum_dim_int_list_31 = _shape_param_15 = None
        convert_element_type_default_31: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_default_32: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_15, torch.float32);  reshape_default_15 = None
        mul_tensor_8: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_63, arg392_1);  arg392_1 = None
        sum_dim_int_list_32: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_33: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_63, [0, 1]);  convert_element_type_63 = None
        sum_dim_int_list_34: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_84, [0], True);  view_84 = None
        reshape_default_16: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_16);  sum_dim_int_list_34 = _shape_param_16 = None
        convert_element_type_default_33: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_default_34: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_16, torch.float32);  reshape_default_16 = None
        sum_dim_int_list_35: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_88, [0], True);  view_88 = None
        reshape_default_17: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_17);  sum_dim_int_list_35 = _shape_param_17 = None
        convert_element_type_default_35: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_default_36: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_17, torch.float32);  reshape_default_17 = None
        mul_tensor_9: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_72, arg396_1);  arg396_1 = None
        sum_dim_int_list_36: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_37: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_72, [0, 1]);  convert_element_type_72 = None
        sum_dim_int_list_38: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_91, [0], True);  view_91 = None
        reshape_default_18: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_18);  sum_dim_int_list_38 = _shape_param_18 = None
        convert_element_type_default_37: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_default_38: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_18, torch.float32);  reshape_default_18 = None
        sum_dim_int_list_39: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_102, [0], True);  view_102 = None
        reshape_default_19: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_19);  sum_dim_int_list_39 = _shape_param_19 = None
        convert_element_type_default_39: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_default_40: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_19, torch.float32);  reshape_default_19 = None
        mul_tensor_10: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_78, arg400_1);  arg400_1 = None
        sum_dim_int_list_40: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_41: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_78, [0, 1]);  convert_element_type_78 = None
        sum_dim_int_list_42: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True);  view_105 = None
        reshape_default_20: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_20);  sum_dim_int_list_42 = _shape_param_20 = None
        convert_element_type_default_41: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_default_42: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_20, torch.float32);  reshape_default_20 = None
        sum_dim_int_list_43: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_109, [0], True);  view_109 = None
        reshape_default_21: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_21);  sum_dim_int_list_43 = _shape_param_21 = None
        convert_element_type_default_43: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_default_44: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_21, torch.float32);  reshape_default_21 = None
        mul_tensor_11: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_87, arg404_1);  arg404_1 = None
        sum_dim_int_list_44: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_45: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_87, [0, 1]);  convert_element_type_87 = None
        sum_dim_int_list_46: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_112, [0], True);  view_112 = None
        reshape_default_22: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_22);  sum_dim_int_list_46 = _shape_param_22 = None
        convert_element_type_default_45: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_default_46: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_22, torch.float32);  reshape_default_22 = None
        sum_dim_int_list_47: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_123, [0], True);  view_123 = None
        reshape_default_23: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_23);  sum_dim_int_list_47 = _shape_param_23 = None
        convert_element_type_default_47: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_default_48: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_23, torch.float32);  reshape_default_23 = None
        mul_tensor_12: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_93, arg408_1);  arg408_1 = None
        sum_dim_int_list_48: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_49: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_93, [0, 1]);  convert_element_type_93 = None
        sum_dim_int_list_50: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
        reshape_default_24: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_24);  sum_dim_int_list_50 = _shape_param_24 = None
        convert_element_type_default_49: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_default_50: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_24, torch.float32);  reshape_default_24 = None
        sum_dim_int_list_51: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_130, [0], True);  view_130 = None
        reshape_default_25: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_25);  sum_dim_int_list_51 = _shape_param_25 = None
        convert_element_type_default_51: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_default_52: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_25, torch.float32);  reshape_default_25 = None
        mul_tensor_13: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_102, arg412_1);  arg412_1 = None
        sum_dim_int_list_52: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_102, [0, 1]);  convert_element_type_102 = None
        sum_dim_int_list_54: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_133, [0], True);  view_133 = None
        reshape_default_26: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, _shape_param_26);  sum_dim_int_list_54 = _shape_param_26 = None
        convert_element_type_default_53: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_default_54: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_26, torch.float32);  reshape_default_26 = None
        sum_dim_int_list_55: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True);  view_144 = None
        reshape_default_27: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_27);  sum_dim_int_list_55 = _shape_param_27 = None
        convert_element_type_default_55: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_default_56: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_27, torch.float32);  reshape_default_27 = None
        mul_tensor_14: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_108, arg416_1);  arg416_1 = None
        sum_dim_int_list_56: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_57: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_108, [0, 1]);  convert_element_type_108 = None
        sum_dim_int_list_58: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_147, [0], True);  view_147 = None
        reshape_default_28: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, _shape_param_28);  sum_dim_int_list_58 = _shape_param_28 = None
        convert_element_type_default_57: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_default_58: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_28, torch.float32);  reshape_default_28 = None
        sum_dim_int_list_59: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_151, [0], True);  view_151 = None
        reshape_default_29: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_29);  sum_dim_int_list_59 = _shape_param_29 = None
        convert_element_type_default_59: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_default_60: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_29, torch.float32);  reshape_default_29 = None
        mul_tensor_15: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_117, arg420_1);  arg420_1 = None
        sum_dim_int_list_60: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_61: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_117, [0, 1]);  convert_element_type_117 = None
        sum_dim_int_list_62: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True);  view_154 = None
        reshape_default_30: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_30);  sum_dim_int_list_62 = _shape_param_30 = None
        convert_element_type_default_61: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_default_62: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_30, torch.float32);  reshape_default_30 = None
        sum_dim_int_list_63: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
        reshape_default_31: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_31);  sum_dim_int_list_63 = _shape_param_31 = None
        convert_element_type_default_63: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_default_64: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_31, torch.float32);  reshape_default_31 = None
        mul_tensor_16: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_123, arg424_1);  arg424_1 = None
        sum_dim_int_list_64: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_65: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_123, [0, 1]);  convert_element_type_123 = None
        sum_dim_int_list_66: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
        reshape_default_32: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_32);  sum_dim_int_list_66 = _shape_param_32 = None
        convert_element_type_default_65: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_default_66: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_32, torch.float32);  reshape_default_32 = None
        sum_dim_int_list_67: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_172, [0], True);  view_172 = None
        reshape_default_33: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_33);  sum_dim_int_list_67 = _shape_param_33 = None
        convert_element_type_default_67: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_default_68: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_33, torch.float32);  reshape_default_33 = None
        mul_tensor_17: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_132, arg428_1);  arg428_1 = None
        sum_dim_int_list_68: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_69: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_132, [0, 1]);  convert_element_type_132 = None
        sum_dim_int_list_70: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_175, [0], True);  view_175 = None
        reshape_default_34: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_34);  sum_dim_int_list_70 = _shape_param_34 = None
        convert_element_type_default_69: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_default_70: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_34, torch.float32);  reshape_default_34 = None
        sum_dim_int_list_71: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        reshape_default_35: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_35);  sum_dim_int_list_71 = _shape_param_35 = None
        convert_element_type_default_71: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_default_72: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_35, torch.float32);  reshape_default_35 = None
        mul_tensor_18: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_138, arg432_1);  arg432_1 = None
        sum_dim_int_list_72: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_73: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_138, [0, 1]);  convert_element_type_138 = None
        sum_dim_int_list_74: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
        reshape_default_36: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_74, _shape_param_36);  sum_dim_int_list_74 = _shape_param_36 = None
        convert_element_type_default_73: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_default_74: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_36, torch.float32);  reshape_default_36 = None
        sum_dim_int_list_75: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_193, [0], True);  view_193 = None
        reshape_default_37: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_37);  sum_dim_int_list_75 = _shape_param_37 = None
        convert_element_type_default_75: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_default_76: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_37, torch.float32);  reshape_default_37 = None
        mul_tensor_19: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_147, arg436_1);  arg436_1 = None
        sum_dim_int_list_76: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_77: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_147, [0, 1]);  convert_element_type_147 = None
        sum_dim_int_list_78: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_196, [0], True);  view_196 = None
        reshape_default_38: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_78, _shape_param_38);  sum_dim_int_list_78 = _shape_param_38 = None
        convert_element_type_default_77: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_default_78: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_38, torch.float32);  reshape_default_38 = None
        sum_dim_int_list_79: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
        reshape_default_39: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_39);  sum_dim_int_list_79 = _shape_param_39 = None
        convert_element_type_default_79: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_default_80: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_39, torch.float32);  reshape_default_39 = None
        mul_tensor_20: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_153, arg440_1);  arg440_1 = None
        sum_dim_int_list_80: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_81: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_153, [0, 1]);  convert_element_type_153 = None
        sum_dim_int_list_82: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
        reshape_default_40: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_40);  sum_dim_int_list_82 = _shape_param_40 = None
        convert_element_type_default_81: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_default_82: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_40, torch.float32);  reshape_default_40 = None
        sum_dim_int_list_83: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True);  view_214 = None
        reshape_default_41: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_41);  sum_dim_int_list_83 = _shape_param_41 = None
        convert_element_type_default_83: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_default_84: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_41, torch.float32);  reshape_default_41 = None
        mul_tensor_21: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_162, arg444_1);  arg444_1 = None
        sum_dim_int_list_84: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_85: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_162, [0, 1]);  convert_element_type_162 = None
        sum_dim_int_list_86: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_217, [0], True);  view_217 = None
        reshape_default_42: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_42);  sum_dim_int_list_86 = _shape_param_42 = None
        convert_element_type_default_85: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_default_86: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_42, torch.float32);  reshape_default_42 = None
        sum_dim_int_list_87: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_228, [0], True);  view_228 = None
        reshape_default_43: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_43);  sum_dim_int_list_87 = _shape_param_43 = None
        convert_element_type_default_87: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_default_88: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_43, torch.float32);  reshape_default_43 = None
        mul_tensor_22: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_168, arg448_1);  arg448_1 = None
        sum_dim_int_list_88: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_89: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_168, [0, 1]);  convert_element_type_168 = None
        sum_dim_int_list_90: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
        reshape_default_44: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_44);  sum_dim_int_list_90 = _shape_param_44 = None
        convert_element_type_default_89: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_default_90: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_44, torch.float32);  reshape_default_44 = None
        sum_dim_int_list_91: "f16[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_235, [0], True);  view_235 = None
        reshape_default_45: "f16[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_45);  sum_dim_int_list_91 = _shape_param_45 = None
        convert_element_type_default_91: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_default_92: "f32[2048]" = torch.ops.prims.convert_element_type.default(reshape_default_45, torch.float32);  reshape_default_45 = None
        mul_tensor_23: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_177, arg452_1);  arg452_1 = None
        sum_dim_int_list_92: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_93: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_177, [0, 1]);  convert_element_type_177 = None
        sum_dim_int_list_94: "f16[1, 512]" = torch.ops.aten.sum.dim_IntList(view_238, [0], True);  view_238 = None
        reshape_default_46: "f16[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_94, _shape_param_46);  sum_dim_int_list_94 = _shape_param_46 = None
        convert_element_type_default_93: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_default_94: "f32[512]" = torch.ops.prims.convert_element_type.default(reshape_default_46, torch.float32);  reshape_default_46 = None
        sum_dim_int_list_95: "f16[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
        reshape_default_47: "f16[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, _shape_param_47);  sum_dim_int_list_95 = _shape_param_47 = None
        reshape_default_48: "f16[77, 32, 512]" = torch.ops.aten.reshape.default(mm_96, _shape_param_48);  mm_96 = _shape_param_48 = None
        convert_element_type_default_95: "f32[77, 32, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_48, torch.float32);  reshape_default_48 = None
        convert_element_type_default_96: "f32[1536, 512]" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_default_97: "f32[1536]" = torch.ops.prims.convert_element_type.default(reshape_default_47, torch.float32);  reshape_default_47 = None
        add_tensor: "f32[32, 77, 512]" = torch.ops.aten.add.Tensor(arg218_1, arg29_1);  arg218_1 = arg29_1 = None
        permute_default: "f32[77, 32, 512]" = torch.ops.aten.permute.default(add_tensor, [1, 0, 2]);  add_tensor = None
        sub_tensor: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(permute_default, arg219_1);  permute_default = arg219_1 = None
        mul_tensor_24: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(sub_tensor, arg220_1);  sub_tensor = None
        mul_tensor_25: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_95, arg30_1);  arg30_1 = None
        mul_tensor_26: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_25, 512)
        sum_dim_int_list_96: "f32[77, 32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [2], True)
        mul_tensor_27: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_25, mul_tensor_24);  mul_tensor_25 = None
        sum_dim_int_list_97: "f32[77, 32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [2], True);  mul_tensor_27 = None
        mul_tensor_28: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_24, sum_dim_int_list_97);  sum_dim_int_list_97 = None
        sub_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_26, sum_dim_int_list_96);  mul_tensor_26 = sum_dim_int_list_96 = None
        sub_tensor_2: "f32[77, 32, 512]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_28);  sub_tensor_1 = mul_tensor_28 = None
        div_tensor: "f32[77, 32, 1]" = torch.ops.aten.div.Tensor(arg220_1, 512);  arg220_1 = None
        mul_tensor_29: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_30: "f32[77, 32, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_95, mul_tensor_24);  mul_tensor_24 = None
        sum_dim_int_list_98: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_99: "f32[512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_default_95, [0, 1]);  convert_element_type_default_95 = None
        add_tensor_1: "f32[77, 32, 512]" = torch.ops.aten.add.Tensor(add_58, mul_tensor_29);  add_58 = mul_tensor_29 = None
        permute_default_1: "f32[32, 77, 512]" = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None
        sum_dim_int_list_100: "f32[1, 77, 512]" = torch.ops.aten.sum.dim_IntList(permute_default_1, [0], True)
        reshape_default_49: "f32[77, 512]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_49);  sum_dim_int_list_100 = _shape_param_49 = None
        eq_scalar: "b8[32, 77]" = torch.ops.aten.eq.Scalar(arg28_1, -1)
        unsqueeze_default: "b8[32, 77, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[32, 77, 512]" = torch.ops.aten.where.self(unsqueeze_default, full, permute_default_1);  unsqueeze_default = full = permute_default_1 = None
        full_default: "f32[49408, 512]" = torch.ops.aten.full.default([49408, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[49408, 512]" = torch.ops.aten.index_put.default(full_default, [arg28_1], where_self, True);  full_default = arg28_1 = where_self = None
        convert_element_type_default_98: "f32[768, 512]" = torch.ops.prims.convert_element_type.default(mm_98, torch.float32);  mm_98 = None
        mul_tensor_31: "f32[32, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_186, arg458_1);  arg458_1 = None
        sum_dim_int_list_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0]);  mul_tensor_31 = None
        sum_dim_int_list_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_186, [0]);  convert_element_type_186 = None
        sum_dim_int_list_103: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_253, [0], True);  view_253 = None
        reshape_default_50: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_50);  sum_dim_int_list_103 = _shape_param_50 = None
        convert_element_type_default_99: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_default_100: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_50, torch.float32);  reshape_default_50 = None
        sum_dim_int_list_104: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_257, [0], True);  view_257 = None
        reshape_default_51: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_51);  sum_dim_int_list_104 = _shape_param_51 = None
        convert_element_type_default_101: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_default_102: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_51, torch.float32);  reshape_default_51 = None
        mul_tensor_32: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_194, arg213_1);  arg213_1 = None
        sum_dim_int_list_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_194, [0, 1]);  convert_element_type_194 = None
        sum_dim_int_list_107: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True);  view_260 = None
        reshape_default_52: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_52);  sum_dim_int_list_107 = _shape_param_52 = None
        convert_element_type_default_103: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_default_104: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_52, torch.float32);  reshape_default_52 = None
        sum_dim_int_list_108: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_270, [0, 1], True);  view_270 = None
        reshape_default_53: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_53);  sum_dim_int_list_108 = _shape_param_53 = None
        convert_element_type_default_105: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        convert_element_type_default_106: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_53, torch.float32);  reshape_default_53 = None
        mul_tensor_33: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_135, arg204_1);  arg204_1 = None
        sum_dim_int_list_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1]);  mul_tensor_33 = None
        sum_dim_int_list_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_135, [0, 1]);  permute_135 = None
        sum_dim_int_list_111: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
        reshape_default_54: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_54);  sum_dim_int_list_111 = _shape_param_54 = None
        convert_element_type_default_107: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_default_108: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_54, torch.float32);  reshape_default_54 = None
        sum_dim_int_list_112: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
        reshape_default_55: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_55);  sum_dim_int_list_112 = _shape_param_55 = None
        convert_element_type_default_109: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_default_110: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_55, torch.float32);  reshape_default_55 = None
        mul_tensor_34: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_209, arg200_1);  arg200_1 = None
        sum_dim_int_list_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1]);  mul_tensor_34 = None
        sum_dim_int_list_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_209, [0, 1]);  convert_element_type_209 = None
        sum_dim_int_list_115: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True);  view_281 = None
        reshape_default_56: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_115, _shape_param_56);  sum_dim_int_list_115 = _shape_param_56 = None
        convert_element_type_default_111: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_default_112: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_56, torch.float32);  reshape_default_56 = None
        sum_dim_int_list_116: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_291, [0, 1], True);  view_291 = None
        reshape_default_57: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, _shape_param_57);  sum_dim_int_list_116 = _shape_param_57 = None
        convert_element_type_default_113: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        convert_element_type_default_114: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_57, torch.float32);  reshape_default_57 = None
        mul_tensor_35: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_147, arg191_1);  arg191_1 = None
        sum_dim_int_list_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1]);  mul_tensor_35 = None
        sum_dim_int_list_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_147, [0, 1]);  permute_147 = None
        sum_dim_int_list_119: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
        reshape_default_58: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_119, _shape_param_58);  sum_dim_int_list_119 = _shape_param_58 = None
        convert_element_type_default_115: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_default_116: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_58, torch.float32);  reshape_default_58 = None
        sum_dim_int_list_120: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
        reshape_default_59: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, _shape_param_59);  sum_dim_int_list_120 = _shape_param_59 = None
        convert_element_type_default_117: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_default_118: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_59, torch.float32);  reshape_default_59 = None
        mul_tensor_36: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_224, arg187_1);  arg187_1 = None
        sum_dim_int_list_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_36, [0, 1]);  mul_tensor_36 = None
        sum_dim_int_list_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_224, [0, 1]);  convert_element_type_224 = None
        sum_dim_int_list_123: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
        reshape_default_60: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_123, _shape_param_60);  sum_dim_int_list_123 = _shape_param_60 = None
        convert_element_type_default_119: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_default_120: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_60, torch.float32);  reshape_default_60 = None
        sum_dim_int_list_124: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_312, [0, 1], True);  view_312 = None
        reshape_default_61: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_124, _shape_param_61);  sum_dim_int_list_124 = _shape_param_61 = None
        convert_element_type_default_121: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_122, torch.float32);  mm_122 = None
        convert_element_type_default_122: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_61, torch.float32);  reshape_default_61 = None
        mul_tensor_37: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_159, arg178_1);  arg178_1 = None
        sum_dim_int_list_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1]);  mul_tensor_37 = None
        sum_dim_int_list_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_159, [0, 1]);  permute_159 = None
        sum_dim_int_list_127: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_316, [0], True);  view_316 = None
        reshape_default_62: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_127, _shape_param_62);  sum_dim_int_list_127 = _shape_param_62 = None
        convert_element_type_default_123: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_default_124: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_62, torch.float32);  reshape_default_62 = None
        sum_dim_int_list_128: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True);  view_320 = None
        reshape_default_63: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_128, _shape_param_63);  sum_dim_int_list_128 = _shape_param_63 = None
        convert_element_type_default_125: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_default_126: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_63, torch.float32);  reshape_default_63 = None
        mul_tensor_38: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_239, arg174_1);  arg174_1 = None
        sum_dim_int_list_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [0, 1]);  mul_tensor_38 = None
        sum_dim_int_list_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_239, [0, 1]);  convert_element_type_239 = None
        sum_dim_int_list_131: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
        reshape_default_64: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_131, _shape_param_64);  sum_dim_int_list_131 = _shape_param_64 = None
        convert_element_type_default_127: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_default_128: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_64, torch.float32);  reshape_default_64 = None
        sum_dim_int_list_132: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_333, [0, 1], True);  view_333 = None
        reshape_default_65: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_132, _shape_param_65);  sum_dim_int_list_132 = _shape_param_65 = None
        convert_element_type_default_129: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        convert_element_type_default_130: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_65, torch.float32);  reshape_default_65 = None
        mul_tensor_39: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_171, arg165_1);  arg165_1 = None
        sum_dim_int_list_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1]);  mul_tensor_39 = None
        sum_dim_int_list_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_171, [0, 1]);  permute_171 = None
        sum_dim_int_list_135: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
        reshape_default_66: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_135, _shape_param_66);  sum_dim_int_list_135 = _shape_param_66 = None
        convert_element_type_default_131: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_default_132: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_66, torch.float32);  reshape_default_66 = None
        sum_dim_int_list_136: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
        reshape_default_67: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_136, _shape_param_67);  sum_dim_int_list_136 = _shape_param_67 = None
        convert_element_type_default_133: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_default_134: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_67, torch.float32);  reshape_default_67 = None
        mul_tensor_40: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_254, arg161_1);  arg161_1 = None
        sum_dim_int_list_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_40, [0, 1]);  mul_tensor_40 = None
        sum_dim_int_list_138: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_254, [0, 1]);  convert_element_type_254 = None
        sum_dim_int_list_139: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True);  view_344 = None
        reshape_default_68: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_139, _shape_param_68);  sum_dim_int_list_139 = _shape_param_68 = None
        convert_element_type_default_135: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_default_136: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_68, torch.float32);  reshape_default_68 = None
        sum_dim_int_list_140: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_354, [0, 1], True);  view_354 = None
        reshape_default_69: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_140, _shape_param_69);  sum_dim_int_list_140 = _shape_param_69 = None
        convert_element_type_default_137: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        convert_element_type_default_138: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_69, torch.float32);  reshape_default_69 = None
        mul_tensor_41: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_183, arg152_1);  arg152_1 = None
        sum_dim_int_list_141: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1]);  mul_tensor_41 = None
        sum_dim_int_list_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_183, [0, 1]);  permute_183 = None
        sum_dim_int_list_143: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
        reshape_default_70: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_143, _shape_param_70);  sum_dim_int_list_143 = _shape_param_70 = None
        convert_element_type_default_139: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_default_140: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_70, torch.float32);  reshape_default_70 = None
        sum_dim_int_list_144: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
        reshape_default_71: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_144, _shape_param_71);  sum_dim_int_list_144 = _shape_param_71 = None
        convert_element_type_default_141: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_default_142: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_71, torch.float32);  reshape_default_71 = None
        mul_tensor_42: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_269, arg148_1);  arg148_1 = None
        sum_dim_int_list_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [0, 1]);  mul_tensor_42 = None
        sum_dim_int_list_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_269, [0, 1]);  convert_element_type_269 = None
        sum_dim_int_list_147: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_365, [0], True);  view_365 = None
        reshape_default_72: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_147, _shape_param_72);  sum_dim_int_list_147 = _shape_param_72 = None
        convert_element_type_default_143: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_default_144: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_72, torch.float32);  reshape_default_72 = None
        sum_dim_int_list_148: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_375, [0, 1], True);  view_375 = None
        reshape_default_73: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_148, _shape_param_73);  sum_dim_int_list_148 = _shape_param_73 = None
        convert_element_type_default_145: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_146, torch.float32);  mm_146 = None
        convert_element_type_default_146: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_73, torch.float32);  reshape_default_73 = None
        mul_tensor_43: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_195, arg139_1);  arg139_1 = None
        sum_dim_int_list_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1]);  mul_tensor_43 = None
        sum_dim_int_list_150: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_195, [0, 1]);  permute_195 = None
        sum_dim_int_list_151: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_379, [0], True);  view_379 = None
        reshape_default_74: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_151, _shape_param_74);  sum_dim_int_list_151 = _shape_param_74 = None
        convert_element_type_default_147: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_default_148: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_74, torch.float32);  reshape_default_74 = None
        sum_dim_int_list_152: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
        reshape_default_75: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_152, _shape_param_75);  sum_dim_int_list_152 = _shape_param_75 = None
        convert_element_type_default_149: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        convert_element_type_default_150: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_75, torch.float32);  reshape_default_75 = None
        mul_tensor_44: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_284, arg135_1);  arg135_1 = None
        sum_dim_int_list_153: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_44, [0, 1]);  mul_tensor_44 = None
        sum_dim_int_list_154: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_284, [0, 1]);  convert_element_type_284 = None
        sum_dim_int_list_155: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
        reshape_default_76: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_155, _shape_param_76);  sum_dim_int_list_155 = _shape_param_76 = None
        convert_element_type_default_151: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_153, torch.float32);  mm_153 = None
        convert_element_type_default_152: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_76, torch.float32);  reshape_default_76 = None
        sum_dim_int_list_156: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_396, [0, 1], True);  view_396 = None
        reshape_default_77: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_156, _shape_param_77);  sum_dim_int_list_156 = _shape_param_77 = None
        convert_element_type_default_153: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        convert_element_type_default_154: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_77, torch.float32);  reshape_default_77 = None
        mul_tensor_45: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_207, arg126_1);  arg126_1 = None
        sum_dim_int_list_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1]);  mul_tensor_45 = None
        sum_dim_int_list_158: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_207, [0, 1]);  permute_207 = None
        sum_dim_int_list_159: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_400, [0], True);  view_400 = None
        reshape_default_78: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_159, _shape_param_78);  sum_dim_int_list_159 = _shape_param_78 = None
        convert_element_type_default_155: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_157, torch.float32);  mm_157 = None
        convert_element_type_default_156: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_78, torch.float32);  reshape_default_78 = None
        sum_dim_int_list_160: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_404, [0], True);  view_404 = None
        reshape_default_79: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_160, _shape_param_79);  sum_dim_int_list_160 = _shape_param_79 = None
        convert_element_type_default_157: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        convert_element_type_default_158: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_79, torch.float32);  reshape_default_79 = None
        mul_tensor_46: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_299, arg122_1);  arg122_1 = None
        sum_dim_int_list_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [0, 1]);  mul_tensor_46 = None
        sum_dim_int_list_162: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_299, [0, 1]);  convert_element_type_299 = None
        sum_dim_int_list_163: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
        reshape_default_80: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_163, _shape_param_80);  sum_dim_int_list_163 = _shape_param_80 = None
        convert_element_type_default_159: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_161, torch.float32);  mm_161 = None
        convert_element_type_default_160: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_80, torch.float32);  reshape_default_80 = None
        sum_dim_int_list_164: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_417, [0, 1], True);  view_417 = None
        reshape_default_81: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_164, _shape_param_81);  sum_dim_int_list_164 = _shape_param_81 = None
        convert_element_type_default_161: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        convert_element_type_default_162: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_81, torch.float32);  reshape_default_81 = None
        mul_tensor_47: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_219, arg113_1);  arg113_1 = None
        sum_dim_int_list_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1]);  mul_tensor_47 = None
        sum_dim_int_list_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_219, [0, 1]);  permute_219 = None
        sum_dim_int_list_167: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_421, [0], True);  view_421 = None
        reshape_default_82: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_167, _shape_param_82);  sum_dim_int_list_167 = _shape_param_82 = None
        convert_element_type_default_163: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None
        convert_element_type_default_164: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_82, torch.float32);  reshape_default_82 = None
        sum_dim_int_list_168: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
        reshape_default_83: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_168, _shape_param_83);  sum_dim_int_list_168 = _shape_param_83 = None
        convert_element_type_default_165: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None
        convert_element_type_default_166: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_83, torch.float32);  reshape_default_83 = None
        mul_tensor_48: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_314, arg109_1);  arg109_1 = None
        sum_dim_int_list_169: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_48, [0, 1]);  mul_tensor_48 = None
        sum_dim_int_list_170: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_314, [0, 1]);  convert_element_type_314 = None
        sum_dim_int_list_171: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_428, [0], True);  view_428 = None
        reshape_default_84: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_171, _shape_param_84);  sum_dim_int_list_171 = _shape_param_84 = None
        convert_element_type_default_167: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_169, torch.float32);  mm_169 = None
        convert_element_type_default_168: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_84, torch.float32);  reshape_default_84 = None
        sum_dim_int_list_172: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_438, [0, 1], True);  view_438 = None
        reshape_default_85: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_172, _shape_param_85);  sum_dim_int_list_172 = _shape_param_85 = None
        convert_element_type_default_169: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_170, torch.float32);  mm_170 = None
        convert_element_type_default_170: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_85, torch.float32);  reshape_default_85 = None
        mul_tensor_49: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_231, arg100_1);  arg100_1 = None
        sum_dim_int_list_173: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_49, [0, 1]);  mul_tensor_49 = None
        sum_dim_int_list_174: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_231, [0, 1]);  permute_231 = None
        sum_dim_int_list_175: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
        reshape_default_86: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_175, _shape_param_86);  sum_dim_int_list_175 = _shape_param_86 = None
        convert_element_type_default_171: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None
        convert_element_type_default_172: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_86, torch.float32);  reshape_default_86 = None
        sum_dim_int_list_176: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_446, [0], True);  view_446 = None
        reshape_default_87: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_176, _shape_param_87);  sum_dim_int_list_176 = _shape_param_87 = None
        convert_element_type_default_173: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None
        convert_element_type_default_174: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_87, torch.float32);  reshape_default_87 = None
        mul_tensor_50: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_329, arg96_1);  arg96_1 = None
        sum_dim_int_list_177: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_50, [0, 1]);  mul_tensor_50 = None
        sum_dim_int_list_178: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_329, [0, 1]);  convert_element_type_329 = None
        sum_dim_int_list_179: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_449, [0], True);  view_449 = None
        reshape_default_88: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_179, _shape_param_88);  sum_dim_int_list_179 = _shape_param_88 = None
        convert_element_type_default_175: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_177, torch.float32);  mm_177 = None
        convert_element_type_default_176: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_88, torch.float32);  reshape_default_88 = None
        sum_dim_int_list_180: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_459, [0, 1], True);  view_459 = None
        reshape_default_89: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_180, _shape_param_89);  sum_dim_int_list_180 = _shape_param_89 = None
        convert_element_type_default_177: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        convert_element_type_default_178: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_89, torch.float32);  reshape_default_89 = None
        mul_tensor_51: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_243, arg87_1);  arg87_1 = None
        sum_dim_int_list_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_51, [0, 1]);  mul_tensor_51 = None
        sum_dim_int_list_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_243, [0, 1]);  permute_243 = None
        sum_dim_int_list_183: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_463, [0], True);  view_463 = None
        reshape_default_90: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_183, _shape_param_90);  sum_dim_int_list_183 = _shape_param_90 = None
        convert_element_type_default_179: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None
        convert_element_type_default_180: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_90, torch.float32);  reshape_default_90 = None
        sum_dim_int_list_184: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
        reshape_default_91: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_184, _shape_param_91);  sum_dim_int_list_184 = _shape_param_91 = None
        convert_element_type_default_181: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        convert_element_type_default_182: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_91, torch.float32);  reshape_default_91 = None
        mul_tensor_52: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_344, arg83_1);  arg83_1 = None
        sum_dim_int_list_185: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_52, [0, 1]);  mul_tensor_52 = None
        sum_dim_int_list_186: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_344, [0, 1]);  convert_element_type_344 = None
        sum_dim_int_list_187: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
        reshape_default_92: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_187, _shape_param_92);  sum_dim_int_list_187 = _shape_param_92 = None
        convert_element_type_default_183: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_185, torch.float32);  mm_185 = None
        convert_element_type_default_184: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_92, torch.float32);  reshape_default_92 = None
        sum_dim_int_list_188: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_480, [0, 1], True);  view_480 = None
        reshape_default_93: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_188, _shape_param_93);  sum_dim_int_list_188 = _shape_param_93 = None
        convert_element_type_default_185: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_186, torch.float32);  mm_186 = None
        convert_element_type_default_186: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_93, torch.float32);  reshape_default_93 = None
        mul_tensor_53: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_255, arg74_1);  arg74_1 = None
        sum_dim_int_list_189: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_53, [0, 1]);  mul_tensor_53 = None
        sum_dim_int_list_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_255, [0, 1]);  permute_255 = None
        sum_dim_int_list_191: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
        reshape_default_94: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_191, _shape_param_94);  sum_dim_int_list_191 = _shape_param_94 = None
        convert_element_type_default_187: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None
        convert_element_type_default_188: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_94, torch.float32);  reshape_default_94 = None
        sum_dim_int_list_192: "f16[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_488, [0], True);  view_488 = None
        reshape_default_95: "f16[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_192, _shape_param_95);  sum_dim_int_list_192 = _shape_param_95 = None
        convert_element_type_default_189: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_191, torch.float32);  mm_191 = None
        convert_element_type_default_190: "f32[3072]" = torch.ops.prims.convert_element_type.default(reshape_default_95, torch.float32);  reshape_default_95 = None
        mul_tensor_54: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_359, arg70_1);  arg70_1 = None
        sum_dim_int_list_193: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [0, 1]);  mul_tensor_54 = None
        sum_dim_int_list_194: "f32[768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_359, [0, 1]);  convert_element_type_359 = None
        sum_dim_int_list_195: "f16[1, 768]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
        reshape_default_96: "f16[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_195, _shape_param_96);  sum_dim_int_list_195 = _shape_param_96 = None
        convert_element_type_default_191: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_193, torch.float32);  mm_193 = None
        convert_element_type_default_192: "f32[768]" = torch.ops.prims.convert_element_type.default(reshape_default_96, torch.float32);  reshape_default_96 = None
        sum_dim_int_list_196: "f16[1, 1, 2304]" = torch.ops.aten.sum.dim_IntList(view_501, [0, 1], True);  view_501 = None
        reshape_default_97: "f16[2304]" = torch.ops.aten.reshape.default(sum_dim_int_list_196, _shape_param_97);  sum_dim_int_list_196 = _shape_param_97 = None
        convert_element_type_default_193: "f32[2304, 768]" = torch.ops.prims.convert_element_type.default(mm_194, torch.float32);  mm_194 = None
        convert_element_type_default_194: "f32[2304]" = torch.ops.prims.convert_element_type.default(reshape_default_97, torch.float32);  reshape_default_97 = None
        mul_tensor_55: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(permute_267, mul_447);  mul_447 = None
        sum_dim_int_list_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_55, [0, 1]);  mul_tensor_55 = None
        sum_dim_int_list_198: "f32[768]" = torch.ops.aten.sum.dim_IntList(permute_267, [0, 1]);  permute_267 = None
        mul_tensor_56: "f32[32, 50, 768]" = torch.ops.aten.mul.Tensor(add_124, mul_445);  mul_445 = None
        sum_dim_int_list_199: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_56, [0, 1]);  mul_tensor_56 = None
        sum_dim_int_list_200: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_124, [0, 1]);  add_124 = None
        sum_dim_int_list_201: "f32[1, 50, 768]" = torch.ops.aten.sum.dim_IntList(mul_456, [0], True)
        reshape_default_98: "f32[50, 768]" = torch.ops.aten.reshape.default(sum_dim_int_list_201, _shape_param_98);  sum_dim_int_list_201 = _shape_param_98 = None
        slice_tensor: "f32[32, 1, 768]" = torch.ops.aten.slice.Tensor(mul_456, 1, 0, 1);  mul_456 = None
        sum_dim_int_list_202: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_tensor, [0], True);  slice_tensor = None
        reshape_default_99: "f32[1, 768]" = torch.ops.aten.reshape.default(sum_dim_int_list_202, _shape_param_99);  sum_dim_int_list_202 = _shape_param_99 = None
        squeeze_dim: "f32[768]" = torch.ops.aten.squeeze.dim(reshape_default_99, 0);  reshape_default_99 = None
        convert_element_type_default_195: "f32[768, 3, 32, 32]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
        _output_to_half_0: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float16);  convert_element_type_default = None
        _output_to_half_1: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list, torch.float16);  sum_dim_int_list = None
        _output_to_half_2: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_1, torch.float16);  sum_dim_int_list_1 = None
        _output_to_half_3: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float16);  convert_element_type_default_1 = None
        _output_to_half_4: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float16);  convert_element_type_default_2 = None
        _output_to_half_5: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float16);  convert_element_type_default_3 = None
        _output_to_half_6: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float16);  convert_element_type_default_4 = None
        _output_to_half_7: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_4, torch.float16);  sum_dim_int_list_4 = None
        _output_to_half_8: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_5, torch.float16);  sum_dim_int_list_5 = None
        _output_to_half_9: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float16);  convert_element_type_default_5 = None
        _output_to_half_10: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_6, torch.float16);  convert_element_type_default_6 = None
        _output_to_half_11: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_7, torch.float16);  convert_element_type_default_7 = None
        _output_to_half_12: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_8, torch.float16);  convert_element_type_default_8 = None
        _output_to_half_13: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_8, torch.float16);  sum_dim_int_list_8 = None
        _output_to_half_14: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_9, torch.float16);  sum_dim_int_list_9 = None
        _output_to_half_15: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_9, torch.float16);  convert_element_type_default_9 = None
        _output_to_half_16: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_10, torch.float16);  convert_element_type_default_10 = None
        _output_to_half_17: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_11, torch.float16);  convert_element_type_default_11 = None
        _output_to_half_18: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_12, torch.float16);  convert_element_type_default_12 = None
        _output_to_half_19: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_12, torch.float16);  sum_dim_int_list_12 = None
        _output_to_half_20: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_13, torch.float16);  sum_dim_int_list_13 = None
        _output_to_half_21: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_13, torch.float16);  convert_element_type_default_13 = None
        _output_to_half_22: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_14, torch.float16);  convert_element_type_default_14 = None
        _output_to_half_23: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_15, torch.float16);  convert_element_type_default_15 = None
        _output_to_half_24: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_16, torch.float16);  convert_element_type_default_16 = None
        _output_to_half_25: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_16, torch.float16);  sum_dim_int_list_16 = None
        _output_to_half_26: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_17, torch.float16);  sum_dim_int_list_17 = None
        _output_to_half_27: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_17, torch.float16);  convert_element_type_default_17 = None
        _output_to_half_28: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_18, torch.float16);  convert_element_type_default_18 = None
        _output_to_half_29: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_19, torch.float16);  convert_element_type_default_19 = None
        _output_to_half_30: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_20, torch.float16);  convert_element_type_default_20 = None
        _output_to_half_31: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_20, torch.float16);  sum_dim_int_list_20 = None
        _output_to_half_32: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_21, torch.float16);  sum_dim_int_list_21 = None
        _output_to_half_33: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_21, torch.float16);  convert_element_type_default_21 = None
        _output_to_half_34: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_22, torch.float16);  convert_element_type_default_22 = None
        _output_to_half_35: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_23, torch.float16);  convert_element_type_default_23 = None
        _output_to_half_36: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_24, torch.float16);  convert_element_type_default_24 = None
        _output_to_half_37: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_24, torch.float16);  sum_dim_int_list_24 = None
        _output_to_half_38: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_25, torch.float16);  sum_dim_int_list_25 = None
        _output_to_half_39: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_25, torch.float16);  convert_element_type_default_25 = None
        _output_to_half_40: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_26, torch.float16);  convert_element_type_default_26 = None
        _output_to_half_41: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_27, torch.float16);  convert_element_type_default_27 = None
        _output_to_half_42: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_28, torch.float16);  convert_element_type_default_28 = None
        _output_to_half_43: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_28, torch.float16);  sum_dim_int_list_28 = None
        _output_to_half_44: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_29, torch.float16);  sum_dim_int_list_29 = None
        _output_to_half_45: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_29, torch.float16);  convert_element_type_default_29 = None
        _output_to_half_46: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_30, torch.float16);  convert_element_type_default_30 = None
        _output_to_half_47: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_31, torch.float16);  convert_element_type_default_31 = None
        _output_to_half_48: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_32, torch.float16);  convert_element_type_default_32 = None
        _output_to_half_49: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_32, torch.float16);  sum_dim_int_list_32 = None
        _output_to_half_50: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_33, torch.float16);  sum_dim_int_list_33 = None
        _output_to_half_51: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_33, torch.float16);  convert_element_type_default_33 = None
        _output_to_half_52: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_34, torch.float16);  convert_element_type_default_34 = None
        _output_to_half_53: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_35, torch.float16);  convert_element_type_default_35 = None
        _output_to_half_54: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_36, torch.float16);  convert_element_type_default_36 = None
        _output_to_half_55: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_36, torch.float16);  sum_dim_int_list_36 = None
        _output_to_half_56: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_37, torch.float16);  sum_dim_int_list_37 = None
        _output_to_half_57: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_37, torch.float16);  convert_element_type_default_37 = None
        _output_to_half_58: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_38, torch.float16);  convert_element_type_default_38 = None
        _output_to_half_59: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_39, torch.float16);  convert_element_type_default_39 = None
        _output_to_half_60: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_40, torch.float16);  convert_element_type_default_40 = None
        _output_to_half_61: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_40, torch.float16);  sum_dim_int_list_40 = None
        _output_to_half_62: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_41, torch.float16);  sum_dim_int_list_41 = None
        _output_to_half_63: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_41, torch.float16);  convert_element_type_default_41 = None
        _output_to_half_64: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_42, torch.float16);  convert_element_type_default_42 = None
        _output_to_half_65: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_43, torch.float16);  convert_element_type_default_43 = None
        _output_to_half_66: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_44, torch.float16);  convert_element_type_default_44 = None
        _output_to_half_67: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_44, torch.float16);  sum_dim_int_list_44 = None
        _output_to_half_68: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_45, torch.float16);  sum_dim_int_list_45 = None
        _output_to_half_69: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_45, torch.float16);  convert_element_type_default_45 = None
        _output_to_half_70: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_46, torch.float16);  convert_element_type_default_46 = None
        _output_to_half_71: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_47, torch.float16);  convert_element_type_default_47 = None
        _output_to_half_72: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_48, torch.float16);  convert_element_type_default_48 = None
        _output_to_half_73: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_48, torch.float16);  sum_dim_int_list_48 = None
        _output_to_half_74: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_49, torch.float16);  sum_dim_int_list_49 = None
        _output_to_half_75: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_49, torch.float16);  convert_element_type_default_49 = None
        _output_to_half_76: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_50, torch.float16);  convert_element_type_default_50 = None
        _output_to_half_77: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_51, torch.float16);  convert_element_type_default_51 = None
        _output_to_half_78: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_52, torch.float16);  convert_element_type_default_52 = None
        _output_to_half_79: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_52, torch.float16);  sum_dim_int_list_52 = None
        _output_to_half_80: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_53, torch.float16);  sum_dim_int_list_53 = None
        _output_to_half_81: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_53, torch.float16);  convert_element_type_default_53 = None
        _output_to_half_82: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_54, torch.float16);  convert_element_type_default_54 = None
        _output_to_half_83: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_55, torch.float16);  convert_element_type_default_55 = None
        _output_to_half_84: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_56, torch.float16);  convert_element_type_default_56 = None
        _output_to_half_85: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_56, torch.float16);  sum_dim_int_list_56 = None
        _output_to_half_86: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_57, torch.float16);  sum_dim_int_list_57 = None
        _output_to_half_87: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_57, torch.float16);  convert_element_type_default_57 = None
        _output_to_half_88: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_58, torch.float16);  convert_element_type_default_58 = None
        _output_to_half_89: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_59, torch.float16);  convert_element_type_default_59 = None
        _output_to_half_90: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_60, torch.float16);  convert_element_type_default_60 = None
        _output_to_half_91: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_60, torch.float16);  sum_dim_int_list_60 = None
        _output_to_half_92: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_61, torch.float16);  sum_dim_int_list_61 = None
        _output_to_half_93: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_61, torch.float16);  convert_element_type_default_61 = None
        _output_to_half_94: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_62, torch.float16);  convert_element_type_default_62 = None
        _output_to_half_95: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_63, torch.float16);  convert_element_type_default_63 = None
        _output_to_half_96: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_64, torch.float16);  convert_element_type_default_64 = None
        _output_to_half_97: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_64, torch.float16);  sum_dim_int_list_64 = None
        _output_to_half_98: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_65, torch.float16);  sum_dim_int_list_65 = None
        _output_to_half_99: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_65, torch.float16);  convert_element_type_default_65 = None
        _output_to_half_100: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_66, torch.float16);  convert_element_type_default_66 = None
        _output_to_half_101: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_67, torch.float16);  convert_element_type_default_67 = None
        _output_to_half_102: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_68, torch.float16);  convert_element_type_default_68 = None
        _output_to_half_103: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_68, torch.float16);  sum_dim_int_list_68 = None
        _output_to_half_104: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_69, torch.float16);  sum_dim_int_list_69 = None
        _output_to_half_105: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_69, torch.float16);  convert_element_type_default_69 = None
        _output_to_half_106: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_70, torch.float16);  convert_element_type_default_70 = None
        _output_to_half_107: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_71, torch.float16);  convert_element_type_default_71 = None
        _output_to_half_108: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_72, torch.float16);  convert_element_type_default_72 = None
        _output_to_half_109: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_72, torch.float16);  sum_dim_int_list_72 = None
        _output_to_half_110: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_73, torch.float16);  sum_dim_int_list_73 = None
        _output_to_half_111: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_73, torch.float16);  convert_element_type_default_73 = None
        _output_to_half_112: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_74, torch.float16);  convert_element_type_default_74 = None
        _output_to_half_113: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_75, torch.float16);  convert_element_type_default_75 = None
        _output_to_half_114: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_76, torch.float16);  convert_element_type_default_76 = None
        _output_to_half_115: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_76, torch.float16);  sum_dim_int_list_76 = None
        _output_to_half_116: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_77, torch.float16);  sum_dim_int_list_77 = None
        _output_to_half_117: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_77, torch.float16);  convert_element_type_default_77 = None
        _output_to_half_118: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_78, torch.float16);  convert_element_type_default_78 = None
        _output_to_half_119: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_79, torch.float16);  convert_element_type_default_79 = None
        _output_to_half_120: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_80, torch.float16);  convert_element_type_default_80 = None
        _output_to_half_121: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_80, torch.float16);  sum_dim_int_list_80 = None
        _output_to_half_122: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_81, torch.float16);  sum_dim_int_list_81 = None
        _output_to_half_123: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_81, torch.float16);  convert_element_type_default_81 = None
        _output_to_half_124: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_82, torch.float16);  convert_element_type_default_82 = None
        _output_to_half_125: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_83, torch.float16);  convert_element_type_default_83 = None
        _output_to_half_126: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_84, torch.float16);  convert_element_type_default_84 = None
        _output_to_half_127: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_84, torch.float16);  sum_dim_int_list_84 = None
        _output_to_half_128: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_85, torch.float16);  sum_dim_int_list_85 = None
        _output_to_half_129: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_85, torch.float16);  convert_element_type_default_85 = None
        _output_to_half_130: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_86, torch.float16);  convert_element_type_default_86 = None
        _output_to_half_131: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_87, torch.float16);  convert_element_type_default_87 = None
        _output_to_half_132: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_88, torch.float16);  convert_element_type_default_88 = None
        _output_to_half_133: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_88, torch.float16);  sum_dim_int_list_88 = None
        _output_to_half_134: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_89, torch.float16);  sum_dim_int_list_89 = None
        _output_to_half_135: "f16[512, 2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_89, torch.float16);  convert_element_type_default_89 = None
        _output_to_half_136: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_90, torch.float16);  convert_element_type_default_90 = None
        _output_to_half_137: "f16[2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_91, torch.float16);  convert_element_type_default_91 = None
        _output_to_half_138: "f16[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_92, torch.float16);  convert_element_type_default_92 = None
        _output_to_half_139: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_92, torch.float16);  sum_dim_int_list_92 = None
        _output_to_half_140: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_93, torch.float16);  sum_dim_int_list_93 = None
        _output_to_half_141: "f16[512, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_93, torch.float16);  convert_element_type_default_93 = None
        _output_to_half_142: "f16[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_94, torch.float16);  convert_element_type_default_94 = None
        _output_to_half_143: "f16[1536, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_96, torch.float16);  convert_element_type_default_96 = None
        _output_to_half_144: "f16[1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_97, torch.float16);  convert_element_type_default_97 = None
        _output_to_half_145: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_98, torch.float16);  sum_dim_int_list_98 = None
        _output_to_half_146: "f16[512]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_99, torch.float16);  sum_dim_int_list_99 = None
        _output_to_half_147: "f16[77, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_49, torch.float16);  reshape_default_49 = None
        _output_to_half_148: "f16[49408, 512]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.float16);  index_put_default = None
        _output_to_half_149: "f16[768, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_98, torch.float16);  convert_element_type_default_98 = None
        _output_to_half_150: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_101, torch.float16);  sum_dim_int_list_101 = None
        _output_to_half_151: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_102, torch.float16);  sum_dim_int_list_102 = None
        _output_to_half_152: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_99, torch.float16);  convert_element_type_default_99 = None
        _output_to_half_153: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_100, torch.float16);  convert_element_type_default_100 = None
        _output_to_half_154: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_101, torch.float16);  convert_element_type_default_101 = None
        _output_to_half_155: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_102, torch.float16);  convert_element_type_default_102 = None
        _output_to_half_156: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_105, torch.float16);  sum_dim_int_list_105 = None
        _output_to_half_157: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_106, torch.float16);  sum_dim_int_list_106 = None
        _output_to_half_158: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_103, torch.float16);  convert_element_type_default_103 = None
        _output_to_half_159: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_104, torch.float16);  convert_element_type_default_104 = None
        _output_to_half_160: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_105, torch.float16);  convert_element_type_default_105 = None
        _output_to_half_161: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_106, torch.float16);  convert_element_type_default_106 = None
        _output_to_half_162: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_109, torch.float16);  sum_dim_int_list_109 = None
        _output_to_half_163: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_110, torch.float16);  sum_dim_int_list_110 = None
        _output_to_half_164: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_107, torch.float16);  convert_element_type_default_107 = None
        _output_to_half_165: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_108, torch.float16);  convert_element_type_default_108 = None
        _output_to_half_166: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_109, torch.float16);  convert_element_type_default_109 = None
        _output_to_half_167: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_110, torch.float16);  convert_element_type_default_110 = None
        _output_to_half_168: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_113, torch.float16);  sum_dim_int_list_113 = None
        _output_to_half_169: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_114, torch.float16);  sum_dim_int_list_114 = None
        _output_to_half_170: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_111, torch.float16);  convert_element_type_default_111 = None
        _output_to_half_171: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_112, torch.float16);  convert_element_type_default_112 = None
        _output_to_half_172: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_113, torch.float16);  convert_element_type_default_113 = None
        _output_to_half_173: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_114, torch.float16);  convert_element_type_default_114 = None
        _output_to_half_174: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_117, torch.float16);  sum_dim_int_list_117 = None
        _output_to_half_175: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_118, torch.float16);  sum_dim_int_list_118 = None
        _output_to_half_176: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_115, torch.float16);  convert_element_type_default_115 = None
        _output_to_half_177: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_116, torch.float16);  convert_element_type_default_116 = None
        _output_to_half_178: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_117, torch.float16);  convert_element_type_default_117 = None
        _output_to_half_179: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_118, torch.float16);  convert_element_type_default_118 = None
        _output_to_half_180: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_121, torch.float16);  sum_dim_int_list_121 = None
        _output_to_half_181: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_122, torch.float16);  sum_dim_int_list_122 = None
        _output_to_half_182: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_119, torch.float16);  convert_element_type_default_119 = None
        _output_to_half_183: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_120, torch.float16);  convert_element_type_default_120 = None
        _output_to_half_184: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_121, torch.float16);  convert_element_type_default_121 = None
        _output_to_half_185: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_122, torch.float16);  convert_element_type_default_122 = None
        _output_to_half_186: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_125, torch.float16);  sum_dim_int_list_125 = None
        _output_to_half_187: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_126, torch.float16);  sum_dim_int_list_126 = None
        _output_to_half_188: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_123, torch.float16);  convert_element_type_default_123 = None
        _output_to_half_189: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_124, torch.float16);  convert_element_type_default_124 = None
        _output_to_half_190: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_125, torch.float16);  convert_element_type_default_125 = None
        _output_to_half_191: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_126, torch.float16);  convert_element_type_default_126 = None
        _output_to_half_192: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_129, torch.float16);  sum_dim_int_list_129 = None
        _output_to_half_193: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_130, torch.float16);  sum_dim_int_list_130 = None
        _output_to_half_194: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_127, torch.float16);  convert_element_type_default_127 = None
        _output_to_half_195: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_128, torch.float16);  convert_element_type_default_128 = None
        _output_to_half_196: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_129, torch.float16);  convert_element_type_default_129 = None
        _output_to_half_197: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_130, torch.float16);  convert_element_type_default_130 = None
        _output_to_half_198: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_133, torch.float16);  sum_dim_int_list_133 = None
        _output_to_half_199: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_134, torch.float16);  sum_dim_int_list_134 = None
        _output_to_half_200: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_131, torch.float16);  convert_element_type_default_131 = None
        _output_to_half_201: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_132, torch.float16);  convert_element_type_default_132 = None
        _output_to_half_202: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_133, torch.float16);  convert_element_type_default_133 = None
        _output_to_half_203: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_134, torch.float16);  convert_element_type_default_134 = None
        _output_to_half_204: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_137, torch.float16);  sum_dim_int_list_137 = None
        _output_to_half_205: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_138, torch.float16);  sum_dim_int_list_138 = None
        _output_to_half_206: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_135, torch.float16);  convert_element_type_default_135 = None
        _output_to_half_207: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_136, torch.float16);  convert_element_type_default_136 = None
        _output_to_half_208: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_137, torch.float16);  convert_element_type_default_137 = None
        _output_to_half_209: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_138, torch.float16);  convert_element_type_default_138 = None
        _output_to_half_210: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_141, torch.float16);  sum_dim_int_list_141 = None
        _output_to_half_211: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_142, torch.float16);  sum_dim_int_list_142 = None
        _output_to_half_212: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_139, torch.float16);  convert_element_type_default_139 = None
        _output_to_half_213: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_140, torch.float16);  convert_element_type_default_140 = None
        _output_to_half_214: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_141, torch.float16);  convert_element_type_default_141 = None
        _output_to_half_215: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_142, torch.float16);  convert_element_type_default_142 = None
        _output_to_half_216: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_145, torch.float16);  sum_dim_int_list_145 = None
        _output_to_half_217: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_146, torch.float16);  sum_dim_int_list_146 = None
        _output_to_half_218: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_143, torch.float16);  convert_element_type_default_143 = None
        _output_to_half_219: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_144, torch.float16);  convert_element_type_default_144 = None
        _output_to_half_220: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_145, torch.float16);  convert_element_type_default_145 = None
        _output_to_half_221: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_146, torch.float16);  convert_element_type_default_146 = None
        _output_to_half_222: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_149, torch.float16);  sum_dim_int_list_149 = None
        _output_to_half_223: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_150, torch.float16);  sum_dim_int_list_150 = None
        _output_to_half_224: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_147, torch.float16);  convert_element_type_default_147 = None
        _output_to_half_225: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_148, torch.float16);  convert_element_type_default_148 = None
        _output_to_half_226: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_149, torch.float16);  convert_element_type_default_149 = None
        _output_to_half_227: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_150, torch.float16);  convert_element_type_default_150 = None
        _output_to_half_228: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_153, torch.float16);  sum_dim_int_list_153 = None
        _output_to_half_229: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_154, torch.float16);  sum_dim_int_list_154 = None
        _output_to_half_230: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_151, torch.float16);  convert_element_type_default_151 = None
        _output_to_half_231: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_152, torch.float16);  convert_element_type_default_152 = None
        _output_to_half_232: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_153, torch.float16);  convert_element_type_default_153 = None
        _output_to_half_233: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_154, torch.float16);  convert_element_type_default_154 = None
        _output_to_half_234: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_157, torch.float16);  sum_dim_int_list_157 = None
        _output_to_half_235: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_158, torch.float16);  sum_dim_int_list_158 = None
        _output_to_half_236: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_155, torch.float16);  convert_element_type_default_155 = None
        _output_to_half_237: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_156, torch.float16);  convert_element_type_default_156 = None
        _output_to_half_238: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_157, torch.float16);  convert_element_type_default_157 = None
        _output_to_half_239: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_158, torch.float16);  convert_element_type_default_158 = None
        _output_to_half_240: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_161, torch.float16);  sum_dim_int_list_161 = None
        _output_to_half_241: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_162, torch.float16);  sum_dim_int_list_162 = None
        _output_to_half_242: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_159, torch.float16);  convert_element_type_default_159 = None
        _output_to_half_243: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_160, torch.float16);  convert_element_type_default_160 = None
        _output_to_half_244: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_161, torch.float16);  convert_element_type_default_161 = None
        _output_to_half_245: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_162, torch.float16);  convert_element_type_default_162 = None
        _output_to_half_246: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_165, torch.float16);  sum_dim_int_list_165 = None
        _output_to_half_247: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_166, torch.float16);  sum_dim_int_list_166 = None
        _output_to_half_248: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_163, torch.float16);  convert_element_type_default_163 = None
        _output_to_half_249: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_164, torch.float16);  convert_element_type_default_164 = None
        _output_to_half_250: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_165, torch.float16);  convert_element_type_default_165 = None
        _output_to_half_251: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_166, torch.float16);  convert_element_type_default_166 = None
        _output_to_half_252: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_169, torch.float16);  sum_dim_int_list_169 = None
        _output_to_half_253: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_170, torch.float16);  sum_dim_int_list_170 = None
        _output_to_half_254: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_167, torch.float16);  convert_element_type_default_167 = None
        _output_to_half_255: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_168, torch.float16);  convert_element_type_default_168 = None
        _output_to_half_256: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_169, torch.float16);  convert_element_type_default_169 = None
        _output_to_half_257: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_170, torch.float16);  convert_element_type_default_170 = None
        _output_to_half_258: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_173, torch.float16);  sum_dim_int_list_173 = None
        _output_to_half_259: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_174, torch.float16);  sum_dim_int_list_174 = None
        _output_to_half_260: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_171, torch.float16);  convert_element_type_default_171 = None
        _output_to_half_261: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_172, torch.float16);  convert_element_type_default_172 = None
        _output_to_half_262: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_173, torch.float16);  convert_element_type_default_173 = None
        _output_to_half_263: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_174, torch.float16);  convert_element_type_default_174 = None
        _output_to_half_264: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_177, torch.float16);  sum_dim_int_list_177 = None
        _output_to_half_265: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_178, torch.float16);  sum_dim_int_list_178 = None
        _output_to_half_266: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_175, torch.float16);  convert_element_type_default_175 = None
        _output_to_half_267: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_176, torch.float16);  convert_element_type_default_176 = None
        _output_to_half_268: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_177, torch.float16);  convert_element_type_default_177 = None
        _output_to_half_269: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_178, torch.float16);  convert_element_type_default_178 = None
        _output_to_half_270: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_181, torch.float16);  sum_dim_int_list_181 = None
        _output_to_half_271: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_182, torch.float16);  sum_dim_int_list_182 = None
        _output_to_half_272: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_179, torch.float16);  convert_element_type_default_179 = None
        _output_to_half_273: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_180, torch.float16);  convert_element_type_default_180 = None
        _output_to_half_274: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_181, torch.float16);  convert_element_type_default_181 = None
        _output_to_half_275: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_182, torch.float16);  convert_element_type_default_182 = None
        _output_to_half_276: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_185, torch.float16);  sum_dim_int_list_185 = None
        _output_to_half_277: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_186, torch.float16);  sum_dim_int_list_186 = None
        _output_to_half_278: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_183, torch.float16);  convert_element_type_default_183 = None
        _output_to_half_279: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_184, torch.float16);  convert_element_type_default_184 = None
        _output_to_half_280: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_185, torch.float16);  convert_element_type_default_185 = None
        _output_to_half_281: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_186, torch.float16);  convert_element_type_default_186 = None
        _output_to_half_282: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_189, torch.float16);  sum_dim_int_list_189 = None
        _output_to_half_283: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_190, torch.float16);  sum_dim_int_list_190 = None
        _output_to_half_284: "f16[768, 3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_187, torch.float16);  convert_element_type_default_187 = None
        _output_to_half_285: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_188, torch.float16);  convert_element_type_default_188 = None
        _output_to_half_286: "f16[3072, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_189, torch.float16);  convert_element_type_default_189 = None
        _output_to_half_287: "f16[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_190, torch.float16);  convert_element_type_default_190 = None
        _output_to_half_288: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_193, torch.float16);  sum_dim_int_list_193 = None
        _output_to_half_289: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_194, torch.float16);  sum_dim_int_list_194 = None
        _output_to_half_290: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_191, torch.float16);  convert_element_type_default_191 = None
        _output_to_half_291: "f16[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_192, torch.float16);  convert_element_type_default_192 = None
        _output_to_half_292: "f16[2304, 768]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_193, torch.float16);  convert_element_type_default_193 = None
        _output_to_half_293: "f16[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_194, torch.float16);  convert_element_type_default_194 = None
        _output_to_half_294: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_197, torch.float16);  sum_dim_int_list_197 = None
        _output_to_half_295: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_198, torch.float16);  sum_dim_int_list_198 = None
        _output_to_half_296: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_199, torch.float16);  sum_dim_int_list_199 = None
        _output_to_half_297: "f16[768]" = torch.ops.prims.convert_element_type.default(sum_dim_int_list_200, torch.float16);  sum_dim_int_list_200 = None
        _output_to_half_298: "f16[50, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_98, torch.float16);  reshape_default_98 = None
        _output_to_half_299: "f16[768]" = torch.ops.prims.convert_element_type.default(squeeze_dim, torch.float16);  squeeze_dim = None
        _output_to_half_300: "f16[768, 3, 32, 32]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_195, torch.float16);  convert_element_type_default_195 = None
        return (_output_to_half_0, _output_to_half_1, _output_to_half_2, _output_to_half_3, _output_to_half_4, _output_to_half_5, _output_to_half_6, _output_to_half_7, _output_to_half_8, _output_to_half_9, _output_to_half_10, _output_to_half_11, _output_to_half_12, _output_to_half_13, _output_to_half_14, _output_to_half_15, _output_to_half_16, _output_to_half_17, _output_to_half_18, _output_to_half_19, _output_to_half_20, _output_to_half_21, _output_to_half_22, _output_to_half_23, _output_to_half_24, _output_to_half_25, _output_to_half_26, _output_to_half_27, _output_to_half_28, _output_to_half_29, _output_to_half_30, _output_to_half_31, _output_to_half_32, _output_to_half_33, _output_to_half_34, _output_to_half_35, _output_to_half_36, _output_to_half_37, _output_to_half_38, _output_to_half_39, _output_to_half_40, _output_to_half_41, _output_to_half_42, _output_to_half_43, _output_to_half_44, _output_to_half_45, _output_to_half_46, _output_to_half_47, _output_to_half_48, _output_to_half_49, _output_to_half_50, _output_to_half_51, _output_to_half_52, _output_to_half_53, _output_to_half_54, _output_to_half_55, _output_to_half_56, _output_to_half_57, _output_to_half_58, _output_to_half_59, _output_to_half_60, _output_to_half_61, _output_to_half_62, _output_to_half_63, _output_to_half_64, _output_to_half_65, _output_to_half_66, _output_to_half_67, _output_to_half_68, _output_to_half_69, _output_to_half_70, _output_to_half_71, _output_to_half_72, _output_to_half_73, _output_to_half_74, _output_to_half_75, _output_to_half_76, _output_to_half_77, _output_to_half_78, _output_to_half_79, _output_to_half_80, _output_to_half_81, _output_to_half_82, _output_to_half_83, _output_to_half_84, _output_to_half_85, _output_to_half_86, _output_to_half_87, _output_to_half_88, _output_to_half_89, _output_to_half_90, _output_to_half_91, _output_to_half_92, _output_to_half_93, _output_to_half_94, _output_to_half_95, _output_to_half_96, _output_to_half_97, _output_to_half_98, _output_to_half_99, _output_to_half_100, _output_to_half_101, _output_to_half_102, _output_to_half_103, _output_to_half_104, _output_to_half_105, _output_to_half_106, _output_to_half_107, _output_to_half_108, _output_to_half_109, _output_to_half_110, _output_to_half_111, _output_to_half_112, _output_to_half_113, _output_to_half_114, _output_to_half_115, _output_to_half_116, _output_to_half_117, _output_to_half_118, _output_to_half_119, _output_to_half_120, _output_to_half_121, _output_to_half_122, _output_to_half_123, _output_to_half_124, _output_to_half_125, _output_to_half_126, _output_to_half_127, _output_to_half_128, _output_to_half_129, _output_to_half_130, _output_to_half_131, _output_to_half_132, _output_to_half_133, _output_to_half_134, _output_to_half_135, _output_to_half_136, _output_to_half_137, _output_to_half_138, _output_to_half_139, _output_to_half_140, _output_to_half_141, _output_to_half_142, _output_to_half_143, _output_to_half_144, _output_to_half_145, _output_to_half_146, _output_to_half_147, _output_to_half_148, _output_to_half_149, _output_to_half_150, _output_to_half_151, _output_to_half_152, _output_to_half_153, _output_to_half_154, _output_to_half_155, _output_to_half_156, _output_to_half_157, _output_to_half_158, _output_to_half_159, _output_to_half_160, _output_to_half_161, _output_to_half_162, _output_to_half_163, _output_to_half_164, _output_to_half_165, _output_to_half_166, _output_to_half_167, _output_to_half_168, _output_to_half_169, _output_to_half_170, _output_to_half_171, _output_to_half_172, _output_to_half_173, _output_to_half_174, _output_to_half_175, _output_to_half_176, _output_to_half_177, _output_to_half_178, _output_to_half_179, _output_to_half_180, _output_to_half_181, _output_to_half_182, _output_to_half_183, _output_to_half_184, _output_to_half_185, _output_to_half_186, _output_to_half_187, _output_to_half_188, _output_to_half_189, _output_to_half_190, _output_to_half_191, _output_to_half_192, _output_to_half_193, _output_to_half_194, _output_to_half_195, _output_to_half_196, _output_to_half_197, _output_to_half_198, _output_to_half_199, _output_to_half_200, _output_to_half_201, _output_to_half_202, _output_to_half_203, _output_to_half_204, _output_to_half_205, _output_to_half_206, _output_to_half_207, _output_to_half_208, _output_to_half_209, _output_to_half_210, _output_to_half_211, _output_to_half_212, _output_to_half_213, _output_to_half_214, _output_to_half_215, _output_to_half_216, _output_to_half_217, _output_to_half_218, _output_to_half_219, _output_to_half_220, _output_to_half_221, _output_to_half_222, _output_to_half_223, _output_to_half_224, _output_to_half_225, _output_to_half_226, _output_to_half_227, _output_to_half_228, _output_to_half_229, _output_to_half_230, _output_to_half_231, _output_to_half_232, _output_to_half_233, _output_to_half_234, _output_to_half_235, _output_to_half_236, _output_to_half_237, _output_to_half_238, _output_to_half_239, _output_to_half_240, _output_to_half_241, _output_to_half_242, _output_to_half_243, _output_to_half_244, _output_to_half_245, _output_to_half_246, _output_to_half_247, _output_to_half_248, _output_to_half_249, _output_to_half_250, _output_to_half_251, _output_to_half_252, _output_to_half_253, _output_to_half_254, _output_to_half_255, _output_to_half_256, _output_to_half_257, _output_to_half_258, _output_to_half_259, _output_to_half_260, _output_to_half_261, _output_to_half_262, _output_to_half_263, _output_to_half_264, _output_to_half_265, _output_to_half_266, _output_to_half_267, _output_to_half_268, _output_to_half_269, _output_to_half_270, _output_to_half_271, _output_to_half_272, _output_to_half_273, _output_to_half_274, _output_to_half_275, _output_to_half_276, _output_to_half_277, _output_to_half_278, _output_to_half_279, _output_to_half_280, _output_to_half_281, _output_to_half_282, _output_to_half_283, _output_to_half_284, _output_to_half_285, _output_to_half_286, _output_to_half_287, _output_to_half_288, _output_to_half_289, _output_to_half_290, _output_to_half_291, _output_to_half_292, _output_to_half_293, _output_to_half_294, _output_to_half_295, _output_to_half_296, _output_to_half_297, _output_to_half_298, _output_to_half_299, _output_to_half_300)


def _default_make_inputs():
    return [
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([32, 77, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 77, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg364_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg368_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg372_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg376_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg380_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg384_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg388_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg392_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg396_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg400_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg404_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg408_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg412_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg416_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg420_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg424_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg428_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg432_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg436_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg440_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg444_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg448_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([77, 32, 512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # arg452_1
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 1536], dtype=torch.float16, device='cuda'),
    torch.randn([2464, 512], dtype=torch.float16, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float16, device='cuda'),
    torch.randn([32, 77, 512], dtype=torch.float32, device='cuda'),
    torch.randn([77, 512], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([77, 32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn(1261568, dtype=torch.float32, device='cuda').as_strided([77, 32, 512], [512, 39424, 1]),  # add_58
    torch.randint(0, 49408, [32, 77], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([768, 512], dtype=torch.float16, device='cuda'),
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_135
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_147
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_159
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_171
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_183
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_195
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_207
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_219
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_231
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_243
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_255
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1600, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float16, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1600, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([50, 32, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1228800, dtype=torch.float32, device='cuda').as_strided([32, 50, 768], [768, 24576, 1]),  # permute_267
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([32, 50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3, 32, 32], dtype=torch.float16, device='cuda'),
    [512],  # _shape_param_0
    [2048],  # _shape_param_1
    [512],  # _shape_param_2
    [1536],  # _shape_param_3
    [512],  # _shape_param_4
    [2048],  # _shape_param_5
    [512],  # _shape_param_6
    [1536],  # _shape_param_7
    [512],  # _shape_param_8
    [2048],  # _shape_param_9
    [512],  # _shape_param_10
    [1536],  # _shape_param_11
    [512],  # _shape_param_12
    [2048],  # _shape_param_13
    [512],  # _shape_param_14
    [1536],  # _shape_param_15
    [512],  # _shape_param_16
    [2048],  # _shape_param_17
    [512],  # _shape_param_18
    [1536],  # _shape_param_19
    [512],  # _shape_param_20
    [2048],  # _shape_param_21
    [512],  # _shape_param_22
    [1536],  # _shape_param_23
    [512],  # _shape_param_24
    [2048],  # _shape_param_25
    [512],  # _shape_param_26
    [1536],  # _shape_param_27
    [512],  # _shape_param_28
    [2048],  # _shape_param_29
    [512],  # _shape_param_30
    [1536],  # _shape_param_31
    [512],  # _shape_param_32
    [2048],  # _shape_param_33
    [512],  # _shape_param_34
    [1536],  # _shape_param_35
    [512],  # _shape_param_36
    [2048],  # _shape_param_37
    [512],  # _shape_param_38
    [1536],  # _shape_param_39
    [512],  # _shape_param_40
    [2048],  # _shape_param_41
    [512],  # _shape_param_42
    [1536],  # _shape_param_43
    [512],  # _shape_param_44
    [2048],  # _shape_param_45
    [512],  # _shape_param_46
    [1536],  # _shape_param_47
    [77, 32, 512],  # _shape_param_48
    [77, 512],  # _shape_param_49
    [768],  # _shape_param_50
    [3072],  # _shape_param_51
    [768],  # _shape_param_52
    [2304],  # _shape_param_53
    [768],  # _shape_param_54
    [3072],  # _shape_param_55
    [768],  # _shape_param_56
    [2304],  # _shape_param_57
    [768],  # _shape_param_58
    [3072],  # _shape_param_59
    [768],  # _shape_param_60
    [2304],  # _shape_param_61
    [768],  # _shape_param_62
    [3072],  # _shape_param_63
    [768],  # _shape_param_64
    [2304],  # _shape_param_65
    [768],  # _shape_param_66
    [3072],  # _shape_param_67
    [768],  # _shape_param_68
    [2304],  # _shape_param_69
    [768],  # _shape_param_70
    [3072],  # _shape_param_71
    [768],  # _shape_param_72
    [2304],  # _shape_param_73
    [768],  # _shape_param_74
    [3072],  # _shape_param_75
    [768],  # _shape_param_76
    [2304],  # _shape_param_77
    [768],  # _shape_param_78
    [3072],  # _shape_param_79
    [768],  # _shape_param_80
    [2304],  # _shape_param_81
    [768],  # _shape_param_82
    [3072],  # _shape_param_83
    [768],  # _shape_param_84
    [2304],  # _shape_param_85
    [768],  # _shape_param_86
    [3072],  # _shape_param_87
    [768],  # _shape_param_88
    [2304],  # _shape_param_89
    [768],  # _shape_param_90
    [3072],  # _shape_param_91
    [768],  # _shape_param_92
    [2304],  # _shape_param_93
    [768],  # _shape_param_94
    [3072],  # _shape_param_95
    [768],  # _shape_param_96
    [2304],  # _shape_param_97
    [50, 768],  # _shape_param_98
    [1, 768],  # _shape_param_99
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
