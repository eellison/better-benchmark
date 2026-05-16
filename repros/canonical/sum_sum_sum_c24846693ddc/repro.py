"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g35
Pattern hash: c24846693ddc
Shape hash: 6b76c4d5
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm: "f16[32128, 512]", arg249_1: "f32[4, 1024, 512]", arg250_1: "f32[4, 1024, 1]", mul_4: "f32[4, 1024, 512]", mm_2: "f16[512, 2048]", mm_4: "f16[2048, 512]", arg243_1: "f32[4, 1024, 512]", arg244_1: "f32[4, 1024, 1]", convert_element_type_13: "f32[4, 1024, 512]", mm_6: "f16[512, 512]", mm_8: "f16[512, 512]", mm_10: "f16[512, 512]", mm_12: "f16[512, 512]", arg236_1: "f32[4, 1024, 512]", arg237_1: "f32[4, 1024, 1]", convert_element_type_25: "f32[4, 1024, 512]", mm_14: "f16[512, 512]", view_35: "f16[4, 8, 1024, 1024]", mm_16: "f16[512, 512]", mm_18: "f16[512, 512]", mm_20: "f16[512, 512]", arg229_1: "f32[4, 1024, 512]", arg230_1: "f32[4, 1024, 1]", add_7: "f32[4, 1024, 512]", mm_22: "f16[512, 2048]", mm_24: "f16[2048, 512]", arg223_1: "f32[4, 1024, 512]", arg224_1: "f32[4, 1024, 1]", convert_element_type_45: "f32[4, 1024, 512]", mm_26: "f16[512, 512]", mm_28: "f16[512, 512]", mm_30: "f16[512, 512]", mm_32: "f16[512, 512]", arg216_1: "f32[4, 1024, 512]", arg217_1: "f32[4, 1024, 1]", convert_element_type_57: "f32[4, 1024, 512]", mm_34: "f16[512, 512]", view_80: "f16[4, 8, 1024, 1024]", mm_36: "f16[512, 512]", mm_38: "f16[512, 512]", mm_40: "f16[512, 512]", arg209_1: "f32[4, 1024, 512]", arg210_1: "f32[4, 1024, 1]", add_18: "f32[4, 1024, 512]", mm_42: "f16[512, 2048]", mm_44: "f16[2048, 512]", arg203_1: "f32[4, 1024, 512]", arg204_1: "f32[4, 1024, 1]", convert_element_type_77: "f32[4, 1024, 512]", mm_46: "f16[512, 512]", mm_48: "f16[512, 512]", mm_50: "f16[512, 512]", mm_52: "f16[512, 512]", arg196_1: "f32[4, 1024, 512]", arg197_1: "f32[4, 1024, 1]", convert_element_type_89: "f32[4, 1024, 512]", mm_54: "f16[512, 512]", view_125: "f16[4, 8, 1024, 1024]", mm_56: "f16[512, 512]", mm_58: "f16[512, 512]", mm_60: "f16[512, 512]", arg189_1: "f32[4, 1024, 512]", arg190_1: "f32[4, 1024, 1]", add_29: "f32[4, 1024, 512]", mm_62: "f16[512, 2048]", mm_64: "f16[2048, 512]", arg183_1: "f32[4, 1024, 512]", arg184_1: "f32[4, 1024, 1]", convert_element_type_109: "f32[4, 1024, 512]", mm_66: "f16[512, 512]", mm_68: "f16[512, 512]", mm_70: "f16[512, 512]", mm_72: "f16[512, 512]", arg176_1: "f32[4, 1024, 512]", arg177_1: "f32[4, 1024, 1]", convert_element_type_121: "f32[4, 1024, 512]", mm_74: "f16[512, 512]", view_170: "f16[4, 8, 1024, 1024]", mm_76: "f16[512, 512]", mm_78: "f16[512, 512]", mm_80: "f16[512, 512]", arg169_1: "f32[4, 1024, 512]", arg170_1: "f32[4, 1024, 1]", add_40: "f32[4, 1024, 512]", mm_82: "f16[512, 2048]", mm_84: "f16[2048, 512]", arg163_1: "f32[4, 1024, 512]", arg164_1: "f32[4, 1024, 1]", convert_element_type_141: "f32[4, 1024, 512]", mm_86: "f16[512, 512]", mm_88: "f16[512, 512]", mm_90: "f16[512, 512]", mm_92: "f16[512, 512]", arg156_1: "f32[4, 1024, 512]", arg157_1: "f32[4, 1024, 1]", convert_element_type_153: "f32[4, 1024, 512]", mm_94: "f16[512, 512]", view_215: "f16[4, 8, 1024, 1024]", mm_96: "f16[512, 512]", mm_98: "f16[512, 512]", mm_100: "f16[512, 512]", arg149_1: "f32[4, 1024, 512]", arg150_1: "f32[4, 1024, 1]", add_51: "f32[4, 1024, 512]", mm_102: "f16[512, 2048]", mm_104: "f16[2048, 512]", arg143_1: "f32[4, 1024, 512]", arg144_1: "f32[4, 1024, 1]", convert_element_type_173: "f32[4, 1024, 512]", mm_106: "f16[512, 512]", mm_108: "f16[512, 512]", mm_110: "f16[512, 512]", mm_112: "f16[512, 512]", arg135_1: "f32[4, 1024, 512]", arg136_1: "f32[4, 1024, 1]", convert_element_type_185: "f32[4, 1024, 512]", mm_114: "f16[512, 512]", view_261: "f16[4, 8, 1024, 1024]", arg128_1: "i64[1024, 1024]", full_1: "f32[]", mm_117: "f16[4096, 512]", mm_116: "f16[512, 512]", mm_119: "f16[4096, 512]", mm_118: "f16[512, 512]", mm_121: "f16[4096, 512]", mm_120: "f16[512, 512]", arg16_1: "f32[512]", arg124_1: "b8[4, 1024, 512]", arg122_1: "f32[4, 1024, 512]", arg125_1: "f32[4, 1024, 1]", add_59: "f32[4, 1024, 512]", arg15_1: "i64[4, 1024]", arg119_1: "f32[4, 1024, 512]", arg120_1: "f32[4, 1024, 1]", mul_265: "f32[4, 1024, 512]", mm_122: "f16[512, 2048]", mm_124: "f16[2048, 512]", arg113_1: "f32[4, 1024, 512]", arg114_1: "f32[4, 1024, 1]", convert_element_type_208: "f32[4, 1024, 512]", mm_126: "f16[512, 512]", view_288: "f16[4, 8, 1024, 1024]", mm_128: "f16[512, 512]", mm_130: "f16[512, 512]", mm_132: "f16[512, 512]", arg106_1: "f32[4, 1024, 512]", arg107_1: "f32[4, 1024, 1]", add_72: "f32[4, 1024, 512]", mm_134: "f16[512, 2048]", mm_136: "f16[2048, 512]", arg100_1: "f32[4, 1024, 512]", arg101_1: "f32[4, 1024, 1]", convert_element_type_228: "f32[4, 1024, 512]", mm_138: "f16[512, 512]", view_314: "f16[4, 8, 1024, 1024]", mm_140: "f16[512, 512]", mm_142: "f16[512, 512]", mm_144: "f16[512, 512]", arg93_1: "f32[4, 1024, 512]", arg94_1: "f32[4, 1024, 1]", add_79: "f32[4, 1024, 512]", mm_146: "f16[512, 2048]", mm_148: "f16[2048, 512]", arg87_1: "f32[4, 1024, 512]", arg88_1: "f32[4, 1024, 1]", convert_element_type_248: "f32[4, 1024, 512]", mm_150: "f16[512, 512]", view_340: "f16[4, 8, 1024, 1024]", mm_152: "f16[512, 512]", mm_154: "f16[512, 512]", mm_156: "f16[512, 512]", arg80_1: "f32[4, 1024, 512]", arg81_1: "f32[4, 1024, 1]", add_86: "f32[4, 1024, 512]", mm_158: "f16[512, 2048]", mm_160: "f16[2048, 512]", arg74_1: "f32[4, 1024, 512]", arg75_1: "f32[4, 1024, 1]", convert_element_type_268: "f32[4, 1024, 512]", mm_162: "f16[512, 512]", view_366: "f16[4, 8, 1024, 1024]", mm_164: "f16[512, 512]", mm_166: "f16[512, 512]", mm_168: "f16[512, 512]", arg67_1: "f32[4, 1024, 512]", arg68_1: "f32[4, 1024, 1]", add_93: "f32[4, 1024, 512]", mm_170: "f16[512, 2048]", mm_172: "f16[2048, 512]", arg61_1: "f32[4, 1024, 512]", arg62_1: "f32[4, 1024, 1]", convert_element_type_288: "f32[4, 1024, 512]", mm_174: "f16[512, 512]", view_392: "f16[4, 8, 1024, 1024]", mm_176: "f16[512, 512]", mm_178: "f16[512, 512]", mm_180: "f16[512, 512]", arg54_1: "f32[4, 1024, 512]", arg55_1: "f32[4, 1024, 1]", add_100: "f32[4, 1024, 512]", mm_182: "f16[512, 2048]", mm_184: "f16[2048, 512]", arg48_1: "f32[4, 1024, 512]", arg49_1: "f32[4, 1024, 1]", convert_element_type_308: "f32[4, 1024, 512]", mm_186: "f16[512, 512]", view_419: "f16[4, 8, 1024, 1024]", arg41_1: "i64[1024, 1024]", mm_189: "f16[4096, 512]", mm_188: "f16[512, 512]", mm_191: "f16[4096, 512]", mm_190: "f16[512, 512]", mm_193: "f16[4096, 512]", mm_192: "f16[512, 512]", arg1_1: "f32[512]", arg37_1: "b8[4, 1024, 512]", arg35_1: "f32[4, 1024, 512]", arg38_1: "f32[4, 1024, 1]", add_104: "f32[4, 1024, 512]", arg0_1: "i64[4, 1024]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[32128, 512]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mul_tensor: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg249_1, arg250_1);  arg249_1 = arg250_1 = None
        mul_tensor_1: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_4, mul_tensor);  mul_4 = mul_tensor = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list, [512]);  sum_dim_int_list = None
        convert_element_type_default_1: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_default_2: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        mul_tensor_2: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg243_1, arg244_1);  arg243_1 = arg244_1 = None
        mul_tensor_3: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_13, mul_tensor_2);  convert_element_type_13 = mul_tensor_2 = None
        sum_dim_int_list_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_1: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, [512]);  sum_dim_int_list_1 = None
        convert_element_type_default_3: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        convert_element_type_default_4: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        convert_element_type_default_5: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        convert_element_type_default_6: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        mul_tensor_4: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg236_1, arg237_1);  arg236_1 = arg237_1 = None
        mul_tensor_5: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_25, mul_tensor_4);  convert_element_type_25 = mul_tensor_4 = None
        sum_dim_int_list_2: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_2: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, [512]);  sum_dim_int_list_2 = None
        convert_element_type_default_7: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        convert_element_type_default_8: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        convert_element_type_default_9: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_16, torch.float32);  mm_16 = None
        convert_element_type_default_10: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_18, torch.float32);  mm_18 = None
        convert_element_type_default_11: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        mul_tensor_6: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg229_1, arg230_1);  arg229_1 = arg230_1 = None
        mul_tensor_7: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_7, mul_tensor_6);  add_7 = mul_tensor_6 = None
        sum_dim_int_list_3: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_3: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, [512]);  sum_dim_int_list_3 = None
        convert_element_type_default_12: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_22, torch.float32);  mm_22 = None
        convert_element_type_default_13: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        mul_tensor_8: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg223_1, arg224_1);  arg223_1 = arg224_1 = None
        mul_tensor_9: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_45, mul_tensor_8);  convert_element_type_45 = mul_tensor_8 = None
        sum_dim_int_list_4: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_4: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, [512]);  sum_dim_int_list_4 = None
        convert_element_type_default_14: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_26, torch.float32);  mm_26 = None
        convert_element_type_default_15: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_28, torch.float32);  mm_28 = None
        convert_element_type_default_16: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_30, torch.float32);  mm_30 = None
        convert_element_type_default_17: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_32, torch.float32);  mm_32 = None
        mul_tensor_10: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg216_1, arg217_1);  arg216_1 = arg217_1 = None
        mul_tensor_11: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_57, mul_tensor_10);  convert_element_type_57 = mul_tensor_10 = None
        sum_dim_int_list_5: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_5: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, [512]);  sum_dim_int_list_5 = None
        convert_element_type_default_18: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        convert_element_type_default_19: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_80, torch.float32);  view_80 = None
        add_tensor: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_8, convert_element_type_default_19);  convert_element_type_default_8 = convert_element_type_default_19 = None
        convert_element_type_default_20: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        convert_element_type_default_21: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_38, torch.float32);  mm_38 = None
        convert_element_type_default_22: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_40, torch.float32);  mm_40 = None
        mul_tensor_12: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg209_1, arg210_1);  arg209_1 = arg210_1 = None
        mul_tensor_13: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_18, mul_tensor_12);  add_18 = mul_tensor_12 = None
        sum_dim_int_list_6: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1], True);  mul_tensor_13 = None
        reshape_default_6: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, [512]);  sum_dim_int_list_6 = None
        convert_element_type_default_23: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_42, torch.float32);  mm_42 = None
        convert_element_type_default_24: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        mul_tensor_14: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg203_1, arg204_1);  arg203_1 = arg204_1 = None
        mul_tensor_15: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_77, mul_tensor_14);  convert_element_type_77 = mul_tensor_14 = None
        sum_dim_int_list_7: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_7: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, [512]);  sum_dim_int_list_7 = None
        convert_element_type_default_25: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        convert_element_type_default_26: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_48, torch.float32);  mm_48 = None
        convert_element_type_default_27: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_50, torch.float32);  mm_50 = None
        convert_element_type_default_28: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_52, torch.float32);  mm_52 = None
        mul_tensor_16: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg196_1, arg197_1);  arg196_1 = arg197_1 = None
        mul_tensor_17: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_89, mul_tensor_16);  convert_element_type_89 = mul_tensor_16 = None
        sum_dim_int_list_8: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_8: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, [512]);  sum_dim_int_list_8 = None
        convert_element_type_default_29: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_54, torch.float32);  mm_54 = None
        convert_element_type_default_30: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None
        add_tensor_1: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_30);  add_tensor = convert_element_type_default_30 = None
        convert_element_type_default_31: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        convert_element_type_default_32: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        convert_element_type_default_33: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        mul_tensor_18: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg189_1, arg190_1);  arg189_1 = arg190_1 = None
        mul_tensor_19: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_29, mul_tensor_18);  add_29 = mul_tensor_18 = None
        sum_dim_int_list_9: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_9: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, [512]);  sum_dim_int_list_9 = None
        convert_element_type_default_34: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_62, torch.float32);  mm_62 = None
        convert_element_type_default_35: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_64, torch.float32);  mm_64 = None
        mul_tensor_20: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg183_1, arg184_1);  arg183_1 = arg184_1 = None
        mul_tensor_21: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_109, mul_tensor_20);  convert_element_type_109 = mul_tensor_20 = None
        sum_dim_int_list_10: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], True);  mul_tensor_21 = None
        reshape_default_10: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, [512]);  sum_dim_int_list_10 = None
        convert_element_type_default_36: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        convert_element_type_default_37: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        convert_element_type_default_38: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        convert_element_type_default_39: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        mul_tensor_22: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg176_1, arg177_1);  arg176_1 = arg177_1 = None
        mul_tensor_23: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_121, mul_tensor_22);  convert_element_type_121 = mul_tensor_22 = None
        sum_dim_int_list_11: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1], True);  mul_tensor_23 = None
        reshape_default_11: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, [512]);  sum_dim_int_list_11 = None
        convert_element_type_default_40: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_74, torch.float32);  mm_74 = None
        convert_element_type_default_41: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_170, torch.float32);  view_170 = None
        add_tensor_2: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_41);  add_tensor_1 = convert_element_type_default_41 = None
        convert_element_type_default_42: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        convert_element_type_default_43: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        convert_element_type_default_44: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        mul_tensor_24: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg169_1, arg170_1);  arg169_1 = arg170_1 = None
        mul_tensor_25: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_40, mul_tensor_24);  add_40 = mul_tensor_24 = None
        sum_dim_int_list_12: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_12: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, [512]);  sum_dim_int_list_12 = None
        convert_element_type_default_45: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        convert_element_type_default_46: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        mul_tensor_26: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg163_1, arg164_1);  arg163_1 = arg164_1 = None
        mul_tensor_27: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_141, mul_tensor_26);  convert_element_type_141 = mul_tensor_26 = None
        sum_dim_int_list_13: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_13: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, [512]);  sum_dim_int_list_13 = None
        convert_element_type_default_47: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_86, torch.float32);  mm_86 = None
        convert_element_type_default_48: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None
        convert_element_type_default_49: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        convert_element_type_default_50: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        mul_tensor_28: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg156_1, arg157_1);  arg156_1 = arg157_1 = None
        mul_tensor_29: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_153, mul_tensor_28);  convert_element_type_153 = mul_tensor_28 = None
        sum_dim_int_list_14: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1], True);  mul_tensor_29 = None
        reshape_default_14: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, [512]);  sum_dim_int_list_14 = None
        convert_element_type_default_51: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        convert_element_type_default_52: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_215, torch.float32);  view_215 = None
        add_tensor_3: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_52);  add_tensor_2 = convert_element_type_default_52 = None
        convert_element_type_default_53: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        convert_element_type_default_54: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_98, torch.float32);  mm_98 = None
        convert_element_type_default_55: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_100, torch.float32);  mm_100 = None
        mul_tensor_30: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg149_1, arg150_1);  arg149_1 = arg150_1 = None
        mul_tensor_31: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_51, mul_tensor_30);  add_51 = mul_tensor_30 = None
        sum_dim_int_list_15: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1], True);  mul_tensor_31 = None
        reshape_default_15: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, [512]);  sum_dim_int_list_15 = None
        convert_element_type_default_56: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        convert_element_type_default_57: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        mul_tensor_32: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg143_1, arg144_1);  arg143_1 = arg144_1 = None
        mul_tensor_33: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_173, mul_tensor_32);  convert_element_type_173 = mul_tensor_32 = None
        sum_dim_int_list_16: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1], True);  mul_tensor_33 = None
        reshape_default_16: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, [512]);  sum_dim_int_list_16 = None
        convert_element_type_default_58: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        convert_element_type_default_59: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        convert_element_type_default_60: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_110, torch.float32);  mm_110 = None
        convert_element_type_default_61: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_112, torch.float32);  mm_112 = None
        mul_tensor_34: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg135_1, arg136_1);  arg135_1 = arg136_1 = None
        mul_tensor_35: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_185, mul_tensor_34);  convert_element_type_185 = mul_tensor_34 = None
        sum_dim_int_list_17: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1], True);  mul_tensor_35 = None
        reshape_default_17: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, [512]);  sum_dim_int_list_17 = None
        convert_element_type_default_62: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        convert_element_type_default_63: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        add_tensor_4: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_3, convert_element_type_default_63);  add_tensor_3 = convert_element_type_default_63 = None
        sum_dim_int_list_18: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_4, [0], True);  add_tensor_4 = None
        squeeze_dim: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_18, 0);  sum_dim_int_list_18 = None
        permute_default: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None
        eq_scalar: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(arg128_1, -1)
        unsqueeze_default: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_default, full_1, permute_default);  unsqueeze_default = permute_default = None
        clone_default: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_self, memory_format = torch.contiguous_format);  where_self = None
        full_default: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default, [arg128_1], clone_default, True);  arg128_1 = clone_default = None
        reshape_default_18: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_117, [4, 1024, 512]);  mm_117 = None
        convert_element_type_default_64: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_18, torch.float32);  reshape_default_18 = None
        convert_element_type_default_65: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        reshape_default_19: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_119, [4, 1024, 512]);  mm_119 = None
        convert_element_type_default_66: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_19, torch.float32);  reshape_default_19 = None
        add_tensor_5: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default_64, convert_element_type_default_66);  convert_element_type_default_64 = convert_element_type_default_66 = None
        convert_element_type_default_67: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None
        reshape_default_20: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_121, [4, 1024, 512]);  mm_121 = None
        convert_element_type_default_68: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_20, torch.float32);  reshape_default_20 = None
        add_tensor_6: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_5, convert_element_type_default_68);  add_tensor_5 = convert_element_type_default_68 = None
        convert_element_type_default_69: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        mul_tensor_36: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_6, arg16_1);  arg16_1 = None
        mul_tensor_37: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg124_1, arg122_1);  arg122_1 = None
        mul_tensor_38: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 1.1111111111111112);  mul_tensor_37 = None
        mul_tensor_39: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_38, arg125_1)
        mul_tensor_40: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_6, mul_tensor_39);  add_tensor_6 = mul_tensor_39 = None
        sum_dim_int_list_19: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_40, [0, 1], True);  mul_tensor_40 = None
        reshape_default_21: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, [512]);  sum_dim_int_list_19 = None
        mul_tensor_41: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_36, mul_tensor_38)
        mul_tensor_42: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_36, arg125_1);  mul_tensor_36 = None
        sum_dim_int_list_20: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [2], True);  mul_tensor_41 = None
        add_tensor_7: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_59, mul_tensor_42);  add_59 = mul_tensor_42 = None
        pow_tensor_scalar: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg125_1, 3);  arg125_1 = None
        mul_scalar: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_20, -0.5);  sum_dim_int_list_20 = None
        mul_tensor_43: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        expand_default: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_43, [4, 1024, 512]);  mul_tensor_43 = None
        div_scalar: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_38, 1.0);  mul_tensor_38 = None
        mul_scalar_1: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_44: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_8: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_7, mul_tensor_44);  add_tensor_7 = mul_tensor_44 = None
        convert_element_type_default_70: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg124_1, torch.float32);  arg124_1 = None
        mul_tensor_45: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_70, 1.1111111111111112);  convert_element_type_default_70 = None
        mul_tensor_46: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_8, mul_tensor_45);  add_tensor_8 = mul_tensor_45 = None
        eq_scalar_1: "b8[4, 1024]" = torch.ops.aten.eq.Scalar(arg15_1, -1)
        unsqueeze_default_1: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[4, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_default_1, full_1, mul_tensor_46);  unsqueeze_default_1 = mul_tensor_46 = None
        full_default_1: "f32[32128, 512]" = torch.ops.aten.full.default([32128, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[32128, 512]" = torch.ops.aten.index_put.default(full_default_1, [arg15_1], where_self_1, True);  arg15_1 = where_self_1 = None
        add_tensor_9: "f32[32128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default, index_put_default_1);  convert_element_type_default = index_put_default_1 = None
        mul_tensor_47: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg119_1, arg120_1);  arg119_1 = arg120_1 = None
        mul_tensor_48: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_265, mul_tensor_47);  mul_265 = mul_tensor_47 = None
        sum_dim_int_list_21: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_48, [0, 1], True);  mul_tensor_48 = None
        reshape_default_22: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, [512]);  sum_dim_int_list_21 = None
        convert_element_type_default_71: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_122, torch.float32);  mm_122 = None
        convert_element_type_default_72: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_124, torch.float32);  mm_124 = None
        mul_tensor_49: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg113_1, arg114_1);  arg113_1 = arg114_1 = None
        mul_tensor_50: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_208, mul_tensor_49);  convert_element_type_208 = mul_tensor_49 = None
        sum_dim_int_list_22: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_50, [0, 1], True);  mul_tensor_50 = None
        reshape_default_23: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, [512]);  sum_dim_int_list_22 = None
        convert_element_type_default_73: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        convert_element_type_default_74: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_288, torch.float32);  view_288 = None
        convert_element_type_default_75: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        convert_element_type_default_76: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        convert_element_type_default_77: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None
        mul_tensor_51: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg106_1, arg107_1);  arg106_1 = arg107_1 = None
        mul_tensor_52: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_72, mul_tensor_51);  add_72 = mul_tensor_51 = None
        sum_dim_int_list_23: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_52, [0, 1], True);  mul_tensor_52 = None
        reshape_default_24: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, [512]);  sum_dim_int_list_23 = None
        convert_element_type_default_78: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_134, torch.float32);  mm_134 = None
        convert_element_type_default_79: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        mul_tensor_53: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg100_1, arg101_1);  arg100_1 = arg101_1 = None
        mul_tensor_54: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_228, mul_tensor_53);  convert_element_type_228 = mul_tensor_53 = None
        sum_dim_int_list_24: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [0, 1], True);  mul_tensor_54 = None
        reshape_default_25: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, [512]);  sum_dim_int_list_24 = None
        convert_element_type_default_80: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        convert_element_type_default_81: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_314, torch.float32);  view_314 = None
        add_tensor_10: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_default_74, convert_element_type_default_81);  convert_element_type_default_74 = convert_element_type_default_81 = None
        convert_element_type_default_82: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        convert_element_type_default_83: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None
        convert_element_type_default_84: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None
        mul_tensor_55: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg93_1, arg94_1);  arg93_1 = arg94_1 = None
        mul_tensor_56: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_79, mul_tensor_55);  add_79 = mul_tensor_55 = None
        sum_dim_int_list_25: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_56, [0, 1], True);  mul_tensor_56 = None
        reshape_default_26: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, [512]);  sum_dim_int_list_25 = None
        convert_element_type_default_85: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_146, torch.float32);  mm_146 = None
        convert_element_type_default_86: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None
        mul_tensor_57: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg87_1, arg88_1);  arg87_1 = arg88_1 = None
        mul_tensor_58: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_248, mul_tensor_57);  convert_element_type_248 = mul_tensor_57 = None
        sum_dim_int_list_26: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_58, [0, 1], True);  mul_tensor_58 = None
        reshape_default_27: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, [512]);  sum_dim_int_list_26 = None
        convert_element_type_default_87: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        convert_element_type_default_88: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_340, torch.float32);  view_340 = None
        add_tensor_11: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_10, convert_element_type_default_88);  add_tensor_10 = convert_element_type_default_88 = None
        convert_element_type_default_89: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        convert_element_type_default_90: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        convert_element_type_default_91: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        mul_tensor_59: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg80_1, arg81_1);  arg80_1 = arg81_1 = None
        mul_tensor_60: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_86, mul_tensor_59);  add_86 = mul_tensor_59 = None
        sum_dim_int_list_27: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_60, [0, 1], True);  mul_tensor_60 = None
        reshape_default_28: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, [512]);  sum_dim_int_list_27 = None
        convert_element_type_default_92: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_158, torch.float32);  mm_158 = None
        convert_element_type_default_93: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        mul_tensor_61: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg74_1, arg75_1);  arg74_1 = arg75_1 = None
        mul_tensor_62: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_268, mul_tensor_61);  convert_element_type_268 = mul_tensor_61 = None
        sum_dim_int_list_28: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_62, [0, 1], True);  mul_tensor_62 = None
        reshape_default_29: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, [512]);  sum_dim_int_list_28 = None
        convert_element_type_default_94: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        convert_element_type_default_95: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_366, torch.float32);  view_366 = None
        add_tensor_12: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_11, convert_element_type_default_95);  add_tensor_11 = convert_element_type_default_95 = None
        convert_element_type_default_96: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        convert_element_type_default_97: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        convert_element_type_default_98: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None
        mul_tensor_63: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg67_1, arg68_1);  arg67_1 = arg68_1 = None
        mul_tensor_64: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_93, mul_tensor_63);  add_93 = mul_tensor_63 = None
        sum_dim_int_list_29: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_64, [0, 1], True);  mul_tensor_64 = None
        reshape_default_30: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, [512]);  sum_dim_int_list_29 = None
        convert_element_type_default_99: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_170, torch.float32);  mm_170 = None
        convert_element_type_default_100: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_172, torch.float32);  mm_172 = None
        mul_tensor_65: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg61_1, arg62_1);  arg61_1 = arg62_1 = None
        mul_tensor_66: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_288, mul_tensor_65);  convert_element_type_288 = mul_tensor_65 = None
        sum_dim_int_list_30: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_66, [0, 1], True);  mul_tensor_66 = None
        reshape_default_31: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, [512]);  sum_dim_int_list_30 = None
        convert_element_type_default_101: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_174, torch.float32);  mm_174 = None
        convert_element_type_default_102: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_392, torch.float32);  view_392 = None
        add_tensor_13: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_12, convert_element_type_default_102);  add_tensor_12 = convert_element_type_default_102 = None
        convert_element_type_default_103: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        convert_element_type_default_104: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        convert_element_type_default_105: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None
        mul_tensor_67: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg54_1, arg55_1);  arg54_1 = arg55_1 = None
        mul_tensor_68: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_100, mul_tensor_67);  add_100 = mul_tensor_67 = None
        sum_dim_int_list_31: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_68, [0, 1], True);  mul_tensor_68 = None
        reshape_default_32: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, [512]);  sum_dim_int_list_31 = None
        convert_element_type_default_106: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_182, torch.float32);  mm_182 = None
        convert_element_type_default_107: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None
        mul_tensor_69: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg48_1, arg49_1);  arg48_1 = arg49_1 = None
        mul_tensor_70: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_308, mul_tensor_69);  convert_element_type_308 = mul_tensor_69 = None
        sum_dim_int_list_32: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_70, [0, 1], True);  mul_tensor_70 = None
        reshape_default_33: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, [512]);  sum_dim_int_list_32 = None
        convert_element_type_default_108: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_186, torch.float32);  mm_186 = None
        convert_element_type_default_109: "f32[4, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None
        add_tensor_14: "f32[4, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_tensor_13, convert_element_type_default_109);  add_tensor_13 = convert_element_type_default_109 = None
        sum_dim_int_list_33: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_tensor_14, [0], True);  add_tensor_14 = None
        squeeze_dim_1: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_33, 0);  sum_dim_int_list_33 = None
        permute_default_1: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None
        eq_scalar_2: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(arg41_1, -1)
        unsqueeze_default_2: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_default_2, full_1, permute_default_1);  unsqueeze_default_2 = permute_default_1 = None
        clone_default_1: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_self_2, memory_format = torch.contiguous_format);  where_self_2 = None
        index_put_default_2: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default, [arg41_1], clone_default_1, True);  full_default = arg41_1 = clone_default_1 = None
        reshape_default_34: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_189, [4, 1024, 512]);  mm_189 = None
        convert_element_type_default_110: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_34, torch.float32);  reshape_default_34 = None
        convert_element_type_default_111: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        reshape_default_35: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_191, [4, 1024, 512]);  mm_191 = None
        convert_element_type_default_112: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_35, torch.float32);  reshape_default_35 = None
        add_tensor_15: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default_110, convert_element_type_default_112);  convert_element_type_default_110 = convert_element_type_default_112 = None
        convert_element_type_default_113: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None
        reshape_default_36: "f16[4, 1024, 512]" = torch.ops.aten.reshape.default(mm_193, [4, 1024, 512]);  mm_193 = None
        convert_element_type_default_114: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(reshape_default_36, torch.float32);  reshape_default_36 = None
        add_tensor_16: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_15, convert_element_type_default_114);  add_tensor_15 = convert_element_type_default_114 = None
        convert_element_type_default_115: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None
        mul_tensor_71: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_16, arg1_1);  arg1_1 = None
        mul_tensor_72: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(arg37_1, arg35_1);  arg35_1 = None
        mul_tensor_73: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 1.1111111111111112);  mul_tensor_72 = None
        mul_tensor_74: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_73, arg38_1)
        mul_tensor_75: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_16, mul_tensor_74);  add_tensor_16 = mul_tensor_74 = None
        sum_dim_int_list_34: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_75, [0, 1], True);  mul_tensor_75 = None
        reshape_default_37: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, [512]);  sum_dim_int_list_34 = None
        mul_tensor_76: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_71, mul_tensor_73)
        mul_tensor_77: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_71, arg38_1);  mul_tensor_71 = None
        sum_dim_int_list_35: "f32[4, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_76, [2], True);  mul_tensor_76 = None
        add_tensor_17: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_104, mul_tensor_77);  add_104 = mul_tensor_77 = None
        pow_tensor_scalar_2: "f32[4, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg38_1, 3);  arg38_1 = None
        mul_scalar_2: "f32[4, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_35, -0.5);  sum_dim_int_list_35 = None
        mul_tensor_78: "f32[4, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar_2, pow_tensor_scalar_2);  mul_scalar_2 = pow_tensor_scalar_2 = None
        expand_default_1: "f32[4, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_78, [4, 1024, 512]);  mul_tensor_78 = None
        div_scalar_1: "f32[4, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default_1, 512);  expand_default_1 = None
        pow_tensor_scalar_3: "f32[4, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_73, 1.0);  mul_tensor_73 = None
        mul_scalar_3: "f32[4, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_3, 2.0);  pow_tensor_scalar_3 = None
        mul_tensor_79: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_3);  div_scalar_1 = mul_scalar_3 = None
        add_tensor_18: "f32[4, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_17, mul_tensor_79);  add_tensor_17 = mul_tensor_79 = None
        convert_element_type_default_116: "f32[4, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg37_1, torch.float32);  arg37_1 = None
        mul_tensor_80: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_116, 1.1111111111111112);  convert_element_type_default_116 = None
        mul_tensor_81: "f32[4, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_18, mul_tensor_80);  add_tensor_18 = mul_tensor_80 = None
        eq_scalar_3: "b8[4, 1024]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default_3: "b8[4, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_3, -1);  eq_scalar_3 = None
        where_self_3: "f32[4, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_default_3, full_1, mul_tensor_81);  unsqueeze_default_3 = full_1 = mul_tensor_81 = None
        index_put_default_3: "f32[32128, 512]" = torch.ops.aten.index_put.default(full_default_1, [arg0_1], where_self_3, True);  full_default_1 = arg0_1 = where_self_3 = None
        add_tensor_19: "f32[32128, 512]" = torch.ops.aten.add.Tensor(add_tensor_9, index_put_default_3);  add_tensor_9 = index_put_default_3 = None
        return (reshape_default, convert_element_type_default_1, convert_element_type_default_2, reshape_default_1, convert_element_type_default_3, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6, reshape_default_2, convert_element_type_default_7, convert_element_type_default_9, convert_element_type_default_10, convert_element_type_default_11, reshape_default_3, convert_element_type_default_12, convert_element_type_default_13, reshape_default_4, convert_element_type_default_14, convert_element_type_default_15, convert_element_type_default_16, convert_element_type_default_17, reshape_default_5, convert_element_type_default_18, convert_element_type_default_20, convert_element_type_default_21, convert_element_type_default_22, reshape_default_6, convert_element_type_default_23, convert_element_type_default_24, reshape_default_7, convert_element_type_default_25, convert_element_type_default_26, convert_element_type_default_27, convert_element_type_default_28, reshape_default_8, convert_element_type_default_29, convert_element_type_default_31, convert_element_type_default_32, convert_element_type_default_33, reshape_default_9, convert_element_type_default_34, convert_element_type_default_35, reshape_default_10, convert_element_type_default_36, convert_element_type_default_37, convert_element_type_default_38, convert_element_type_default_39, reshape_default_11, convert_element_type_default_40, convert_element_type_default_42, convert_element_type_default_43, convert_element_type_default_44, reshape_default_12, convert_element_type_default_45, convert_element_type_default_46, reshape_default_13, convert_element_type_default_47, convert_element_type_default_48, convert_element_type_default_49, convert_element_type_default_50, reshape_default_14, convert_element_type_default_51, convert_element_type_default_53, convert_element_type_default_54, convert_element_type_default_55, reshape_default_15, convert_element_type_default_56, convert_element_type_default_57, reshape_default_16, convert_element_type_default_58, convert_element_type_default_59, convert_element_type_default_60, convert_element_type_default_61, reshape_default_17, convert_element_type_default_62, index_put_default, convert_element_type_default_65, convert_element_type_default_67, convert_element_type_default_69, reshape_default_21, reshape_default_22, convert_element_type_default_71, convert_element_type_default_72, reshape_default_23, convert_element_type_default_73, convert_element_type_default_75, convert_element_type_default_76, convert_element_type_default_77, reshape_default_24, convert_element_type_default_78, convert_element_type_default_79, reshape_default_25, convert_element_type_default_80, convert_element_type_default_82, convert_element_type_default_83, convert_element_type_default_84, reshape_default_26, convert_element_type_default_85, convert_element_type_default_86, reshape_default_27, convert_element_type_default_87, convert_element_type_default_89, convert_element_type_default_90, convert_element_type_default_91, reshape_default_28, convert_element_type_default_92, convert_element_type_default_93, reshape_default_29, convert_element_type_default_94, convert_element_type_default_96, convert_element_type_default_97, convert_element_type_default_98, reshape_default_30, convert_element_type_default_99, convert_element_type_default_100, reshape_default_31, convert_element_type_default_101, convert_element_type_default_103, convert_element_type_default_104, convert_element_type_default_105, reshape_default_32, convert_element_type_default_106, convert_element_type_default_107, reshape_default_33, convert_element_type_default_108, index_put_default_2, convert_element_type_default_111, convert_element_type_default_113, convert_element_type_default_115, reshape_default_37, add_tensor_19)


def _default_make_inputs():
    return [
    torch.randn([32128, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [1024, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1024, 512], dtype=torch.bool, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float16, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4, 8, 1024, 1024], dtype=torch.float16, device='cuda'),
    torch.randint(0, 2, [1024, 1024], dtype=torch.int64, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([4096, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512, 512], dtype=torch.float16, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1024, 512], dtype=torch.bool, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [4, 1024], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
