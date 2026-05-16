"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g53
Pattern hash: 44f42fa7ef14
Shape hash: 9124564c
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, div: "f16[8, 1000]", view_1: "f16[1584, 768]", view_5: "f16[1584, 3072]", view_8: "f16[1584, 768]", arg178_1: "f16[8, 12, 198, 64]", view_16: "f16[1584, 2304]", view_19: "f16[1584, 768]", view_23: "f16[1584, 3072]", view_26: "f16[1584, 768]", arg165_1: "f16[8, 12, 198, 64]", view_34: "f16[1584, 2304]", view_37: "f16[1584, 768]", view_41: "f16[1584, 3072]", view_44: "f16[1584, 768]", arg152_1: "f16[8, 12, 198, 64]", view_52: "f16[1584, 2304]", view_55: "f16[1584, 768]", view_59: "f16[1584, 3072]", view_62: "f16[1584, 768]", arg139_1: "f16[8, 12, 198, 64]", view_70: "f16[1584, 2304]", view_73: "f16[1584, 768]", view_77: "f16[1584, 3072]", view_80: "f16[1584, 768]", arg126_1: "f16[8, 12, 198, 64]", view_88: "f16[1584, 2304]", view_91: "f16[1584, 768]", view_95: "f16[1584, 3072]", view_98: "f16[1584, 768]", arg113_1: "f16[8, 12, 198, 64]", view_106: "f16[1584, 2304]", view_109: "f16[1584, 768]", view_113: "f16[1584, 3072]", view_116: "f16[1584, 768]", arg100_1: "f16[8, 12, 198, 64]", view_124: "f16[1584, 2304]", view_127: "f16[1584, 768]", view_131: "f16[1584, 3072]", view_134: "f16[1584, 768]", arg87_1: "f16[8, 12, 198, 64]", view_142: "f16[1584, 2304]", view_145: "f16[1584, 768]", view_149: "f16[1584, 3072]", view_152: "f16[1584, 768]", arg74_1: "f16[8, 12, 198, 64]", view_160: "f16[1584, 2304]", view_163: "f16[1584, 768]", view_167: "f16[1584, 3072]", view_170: "f16[1584, 768]", arg61_1: "f16[8, 12, 198, 64]", view_178: "f16[1584, 2304]", view_181: "f16[1584, 768]", view_185: "f16[1584, 3072]", view_188: "f16[1584, 768]", arg48_1: "f16[8, 12, 198, 64]", view_196: "f16[1584, 2304]", view_199: "f16[1584, 768]", view_203: "f16[1584, 3072]", view_206: "f16[1584, 768]", arg35_1: "f16[8, 12, 198, 64]", view_214: "f16[1584, 2304]", mm_98: "f16[1584, 768]", arg1_1: "f32[768]", arg28_1: "f32[8, 198, 768]", arg0_1: "f32[1, 198, 768]", arg29_1: "f32[8, 198, 1]", arg30_1: "f32[8, 198, 1]", add_47: "f32[8, 198, 768]"):
        # No stacktrace found for following nodes
        permute_default: "f16[1000, 8]" = torch.ops.aten.permute.default(div, [1, 0]);  div = None
        permute_default_1: "f16[768, 1584]" = torch.ops.aten.permute.default(view_1, [1, 0]);  view_1 = None
        permute_default_2: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_5, [1, 0]);  view_5 = None
        permute_default_3: "f16[768, 1584]" = torch.ops.aten.permute.default(view_8, [1, 0]);  view_8 = None
        permute_default_4: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg178_1, [0, 2, 1, 3]);  arg178_1 = None
        reshape_default: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_4, [8, 198, 768]);  permute_default_4 = None
        reshape_default_1: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default, [1584, 768]);  reshape_default = None
        permute_default_5: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_16, [1, 0]);  view_16 = None
        permute_default_6: "f16[768, 1584]" = torch.ops.aten.permute.default(view_19, [1, 0]);  view_19 = None
        permute_default_7: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_23, [1, 0]);  view_23 = None
        permute_default_8: "f16[768, 1584]" = torch.ops.aten.permute.default(view_26, [1, 0]);  view_26 = None
        permute_default_9: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg165_1, [0, 2, 1, 3]);  arg165_1 = None
        reshape_default_2: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_9, [8, 198, 768]);  permute_default_9 = None
        reshape_default_3: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default_2, [1584, 768]);  reshape_default_2 = None
        permute_default_10: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_34, [1, 0]);  view_34 = None
        permute_default_11: "f16[768, 1584]" = torch.ops.aten.permute.default(view_37, [1, 0]);  view_37 = None
        permute_default_12: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_41, [1, 0]);  view_41 = None
        permute_default_13: "f16[768, 1584]" = torch.ops.aten.permute.default(view_44, [1, 0]);  view_44 = None
        permute_default_14: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg152_1, [0, 2, 1, 3]);  arg152_1 = None
        reshape_default_4: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_14, [8, 198, 768]);  permute_default_14 = None
        reshape_default_5: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default_4, [1584, 768]);  reshape_default_4 = None
        permute_default_15: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_52, [1, 0]);  view_52 = None
        permute_default_16: "f16[768, 1584]" = torch.ops.aten.permute.default(view_55, [1, 0]);  view_55 = None
        permute_default_17: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_59, [1, 0]);  view_59 = None
        permute_default_18: "f16[768, 1584]" = torch.ops.aten.permute.default(view_62, [1, 0]);  view_62 = None
        permute_default_19: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg139_1, [0, 2, 1, 3]);  arg139_1 = None
        reshape_default_6: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_19, [8, 198, 768]);  permute_default_19 = None
        reshape_default_7: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default_6, [1584, 768]);  reshape_default_6 = None
        permute_default_20: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_70, [1, 0]);  view_70 = None
        permute_default_21: "f16[768, 1584]" = torch.ops.aten.permute.default(view_73, [1, 0]);  view_73 = None
        permute_default_22: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_77, [1, 0]);  view_77 = None
        permute_default_23: "f16[768, 1584]" = torch.ops.aten.permute.default(view_80, [1, 0]);  view_80 = None
        permute_default_24: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg126_1, [0, 2, 1, 3]);  arg126_1 = None
        reshape_default_8: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_24, [8, 198, 768]);  permute_default_24 = None
        reshape_default_9: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default_8, [1584, 768]);  reshape_default_8 = None
        permute_default_25: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_88, [1, 0]);  view_88 = None
        permute_default_26: "f16[768, 1584]" = torch.ops.aten.permute.default(view_91, [1, 0]);  view_91 = None
        permute_default_27: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_95, [1, 0]);  view_95 = None
        permute_default_28: "f16[768, 1584]" = torch.ops.aten.permute.default(view_98, [1, 0]);  view_98 = None
        permute_default_29: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg113_1, [0, 2, 1, 3]);  arg113_1 = None
        reshape_default_10: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_29, [8, 198, 768]);  permute_default_29 = None
        reshape_default_11: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default_10, [1584, 768]);  reshape_default_10 = None
        permute_default_30: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_106, [1, 0]);  view_106 = None
        permute_default_31: "f16[768, 1584]" = torch.ops.aten.permute.default(view_109, [1, 0]);  view_109 = None
        permute_default_32: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_113, [1, 0]);  view_113 = None
        permute_default_33: "f16[768, 1584]" = torch.ops.aten.permute.default(view_116, [1, 0]);  view_116 = None
        permute_default_34: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg100_1, [0, 2, 1, 3]);  arg100_1 = None
        reshape_default_12: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_34, [8, 198, 768]);  permute_default_34 = None
        reshape_default_13: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default_12, [1584, 768]);  reshape_default_12 = None
        permute_default_35: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_124, [1, 0]);  view_124 = None
        permute_default_36: "f16[768, 1584]" = torch.ops.aten.permute.default(view_127, [1, 0]);  view_127 = None
        permute_default_37: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_131, [1, 0]);  view_131 = None
        permute_default_38: "f16[768, 1584]" = torch.ops.aten.permute.default(view_134, [1, 0]);  view_134 = None
        permute_default_39: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg87_1, [0, 2, 1, 3]);  arg87_1 = None
        reshape_default_14: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_39, [8, 198, 768]);  permute_default_39 = None
        reshape_default_15: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default_14, [1584, 768]);  reshape_default_14 = None
        permute_default_40: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_142, [1, 0]);  view_142 = None
        permute_default_41: "f16[768, 1584]" = torch.ops.aten.permute.default(view_145, [1, 0]);  view_145 = None
        permute_default_42: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_149, [1, 0]);  view_149 = None
        permute_default_43: "f16[768, 1584]" = torch.ops.aten.permute.default(view_152, [1, 0]);  view_152 = None
        permute_default_44: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg74_1, [0, 2, 1, 3]);  arg74_1 = None
        reshape_default_16: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_44, [8, 198, 768]);  permute_default_44 = None
        reshape_default_17: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default_16, [1584, 768]);  reshape_default_16 = None
        permute_default_45: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_160, [1, 0]);  view_160 = None
        permute_default_46: "f16[768, 1584]" = torch.ops.aten.permute.default(view_163, [1, 0]);  view_163 = None
        permute_default_47: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_167, [1, 0]);  view_167 = None
        permute_default_48: "f16[768, 1584]" = torch.ops.aten.permute.default(view_170, [1, 0]);  view_170 = None
        permute_default_49: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg61_1, [0, 2, 1, 3]);  arg61_1 = None
        reshape_default_18: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_49, [8, 198, 768]);  permute_default_49 = None
        reshape_default_19: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default_18, [1584, 768]);  reshape_default_18 = None
        permute_default_50: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_178, [1, 0]);  view_178 = None
        permute_default_51: "f16[768, 1584]" = torch.ops.aten.permute.default(view_181, [1, 0]);  view_181 = None
        permute_default_52: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_185, [1, 0]);  view_185 = None
        permute_default_53: "f16[768, 1584]" = torch.ops.aten.permute.default(view_188, [1, 0]);  view_188 = None
        permute_default_54: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg48_1, [0, 2, 1, 3]);  arg48_1 = None
        reshape_default_20: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_54, [8, 198, 768]);  permute_default_54 = None
        reshape_default_21: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default_20, [1584, 768]);  reshape_default_20 = None
        permute_default_55: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_196, [1, 0]);  view_196 = None
        permute_default_56: "f16[768, 1584]" = torch.ops.aten.permute.default(view_199, [1, 0]);  view_199 = None
        permute_default_57: "f16[3072, 1584]" = torch.ops.aten.permute.default(view_203, [1, 0]);  view_203 = None
        permute_default_58: "f16[768, 1584]" = torch.ops.aten.permute.default(view_206, [1, 0]);  view_206 = None
        permute_default_59: "f16[8, 198, 12, 64]" = torch.ops.aten.permute.default(arg35_1, [0, 2, 1, 3]);  arg35_1 = None
        reshape_default_22: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(permute_default_59, [8, 198, 768]);  permute_default_59 = None
        reshape_default_23: "f16[1584, 768]" = torch.ops.aten.reshape.default(reshape_default_22, [1584, 768]);  reshape_default_22 = None
        permute_default_60: "f16[2304, 1584]" = torch.ops.aten.permute.default(view_214, [1, 0]);  view_214 = None
        reshape_default_24: "f16[8, 198, 768]" = torch.ops.aten.reshape.default(mm_98, [8, 198, 768]);  mm_98 = None
        convert_element_type_default: "f32[8, 198, 768]" = torch.ops.prims.convert_element_type.default(reshape_default_24, torch.float32);  reshape_default_24 = None
        mul_tensor: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, arg1_1);  convert_element_type_default = arg1_1 = None
        mul_tensor_1: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        add_tensor: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(arg28_1, arg0_1);  arg28_1 = arg0_1 = None
        sub_tensor: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_tensor, arg29_1);  add_tensor = arg29_1 = None
        mul_tensor_2: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, arg30_1);  sub_tensor = None
        mul_tensor_3: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        mul_tensor_4: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(arg30_1, 768);  arg30_1 = None
        mul_tensor_5: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_47, mul_tensor_5);  add_47 = mul_tensor_5 = None
        slice_tensor: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(add_tensor_1, 1, 2, 198);  add_tensor_1 = None
        convert_element_type_default_1: "f16[8, 196, 768]" = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float16);  slice_tensor = None
        permute_default_61: "f16[8, 768, 196]" = torch.ops.aten.permute.default(convert_element_type_default_1, [0, 2, 1]);  convert_element_type_default_1 = None
        reshape_default_25: "f16[8, 768, 14, 14]" = torch.ops.aten.reshape.default(permute_default_61, [8, 768, 14, 14]);  permute_default_61 = None
        return (permute_default, permute_default_1, permute_default_2, permute_default_3, reshape_default_1, permute_default_5, permute_default_6, permute_default_7, permute_default_8, reshape_default_3, permute_default_10, permute_default_11, permute_default_12, permute_default_13, reshape_default_5, permute_default_15, permute_default_16, permute_default_17, permute_default_18, reshape_default_7, permute_default_20, permute_default_21, permute_default_22, permute_default_23, reshape_default_9, permute_default_25, permute_default_26, permute_default_27, permute_default_28, reshape_default_11, permute_default_30, permute_default_31, permute_default_32, permute_default_33, reshape_default_13, permute_default_35, permute_default_36, permute_default_37, permute_default_38, reshape_default_15, permute_default_40, permute_default_41, permute_default_42, permute_default_43, reshape_default_17, permute_default_45, permute_default_46, permute_default_47, permute_default_48, reshape_default_19, permute_default_50, permute_default_51, permute_default_52, permute_default_53, reshape_default_21, permute_default_55, permute_default_56, permute_default_57, permute_default_58, reshape_default_23, permute_default_60, reshape_default_25)


def _default_make_inputs():
    return [
    torch.randn([8, 1000], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg178_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg165_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg152_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg139_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg126_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg113_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg100_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg87_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg74_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg61_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg48_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 3072], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn(1216512, dtype=torch.float16, device='cuda').as_strided([8, 12, 198, 64], [152064, 64, 768, 1]),  # arg35_1
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 1], dtype=torch.float32, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
