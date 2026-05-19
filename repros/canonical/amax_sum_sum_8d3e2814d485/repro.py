"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-5-5-linux.aws.a100_graph11
Pattern hash: 8d3e2814d485
Shape hash: 61961b30
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, embedding: "bf16[512, 1, 1024]", amax: "f32[1, 16, 512, 1]", exp: "f32[1, 16, 512, 512]", rsqrt: "f32[512, 1, 1]", rsqrt_1: "f32[512, 1, 1]", convert_element_type_10: "bf16[512, 1, 1024]", amax_1: "f32[1, 16, 512, 1]", exp_1: "f32[1, 16, 512, 512]", rsqrt_2: "f32[512, 1, 1]", rsqrt_3: "f32[512, 1, 1]", convert_element_type_18: "bf16[512, 1, 1024]", amax_2: "f32[1, 16, 512, 1]", exp_2: "f32[1, 16, 512, 512]", rsqrt_4: "f32[512, 1, 1]", rsqrt_5: "f32[512, 1, 1]", convert_element_type_26: "bf16[512, 1, 1024]", amax_3: "f32[1, 16, 512, 1]", exp_3: "f32[1, 16, 512, 512]", rsqrt_6: "f32[512, 1, 1]", rsqrt_7: "f32[512, 1, 1]", convert_element_type_34: "bf16[512, 1, 1024]", amax_4: "f32[1, 16, 512, 1]", exp_4: "f32[1, 16, 512, 512]", rsqrt_8: "f32[512, 1, 1]", rsqrt_9: "f32[512, 1, 1]", convert_element_type_42: "bf16[512, 1, 1024]", amax_5: "f32[1, 16, 512, 1]", exp_5: "f32[1, 16, 512, 512]", rsqrt_10: "f32[512, 1, 1]", rsqrt_11: "f32[512, 1, 1]", convert_element_type_50: "bf16[512, 1, 1024]", amax_6: "f32[1, 16, 512, 1]", exp_6: "f32[1, 16, 512, 512]", rsqrt_12: "f32[512, 1, 1]", rsqrt_13: "f32[512, 1, 1]", convert_element_type_58: "bf16[512, 1, 1024]", amax_7: "f32[1, 16, 512, 1]", exp_7: "f32[1, 16, 512, 512]", rsqrt_14: "f32[512, 1, 1]", rsqrt_15: "f32[512, 1, 1]", convert_element_type_66: "bf16[512, 1, 1024]", amax_8: "f32[1, 16, 512, 1]", exp_8: "f32[1, 16, 512, 512]", rsqrt_16: "f32[512, 1, 1]", rsqrt_17: "f32[512, 1, 1]", convert_element_type_74: "bf16[512, 1, 1024]", amax_9: "f32[1, 16, 512, 1]", exp_9: "f32[1, 16, 512, 512]", rsqrt_18: "f32[512, 1, 1]", rsqrt_19: "f32[512, 1, 1]", convert_element_type_82: "bf16[512, 1, 1024]", amax_10: "f32[1, 16, 512, 1]", exp_10: "f32[1, 16, 512, 512]", rsqrt_20: "f32[512, 1, 1]", rsqrt_21: "f32[512, 1, 1]", convert_element_type_90: "bf16[512, 1, 1024]", amax_11: "f32[1, 16, 512, 1]", exp_11: "f32[1, 16, 512, 512]", rsqrt_22: "f32[512, 1, 1]", rsqrt_23: "f32[512, 1, 1]", convert_element_type_98: "bf16[512, 1, 1024]", amax_12: "f32[1, 16, 512, 1]", exp_12: "f32[1, 16, 512, 512]", rsqrt_24: "f32[512, 1, 1]", rsqrt_25: "f32[512, 1, 1]", convert_element_type_106: "bf16[512, 1, 1024]", amax_13: "f32[1, 16, 512, 1]", exp_13: "f32[1, 16, 512, 512]", rsqrt_26: "f32[512, 1, 1]", rsqrt_27: "f32[512, 1, 1]", convert_element_type_114: "bf16[512, 1, 1024]", amax_14: "f32[1, 16, 512, 1]", exp_14: "f32[1, 16, 512, 512]", rsqrt_28: "f32[512, 1, 1]", rsqrt_29: "f32[512, 1, 1]", convert_element_type_122: "bf16[512, 1, 1024]", amax_15: "f32[1, 16, 512, 1]", exp_15: "f32[1, 16, 512, 512]", rsqrt_30: "f32[512, 1, 1]", rsqrt_31: "f32[512, 1, 1]", convert_element_type_130: "bf16[512, 1, 1024]", amax_16: "f32[1, 16, 512, 1]", exp_16: "f32[1, 16, 512, 512]", rsqrt_32: "f32[512, 1, 1]", rsqrt_33: "f32[512, 1, 1]", convert_element_type_138: "bf16[512, 1, 1024]", amax_17: "f32[1, 16, 512, 1]", exp_17: "f32[1, 16, 512, 512]", rsqrt_34: "f32[512, 1, 1]", rsqrt_35: "f32[512, 1, 1]", convert_element_type_146: "bf16[512, 1, 1024]", amax_18: "f32[1, 16, 512, 1]", exp_18: "f32[1, 16, 512, 512]", rsqrt_36: "f32[512, 1, 1]", rsqrt_37: "f32[512, 1, 1]", convert_element_type_154: "bf16[512, 1, 1024]", amax_19: "f32[1, 16, 512, 1]", exp_19: "f32[1, 16, 512, 512]", rsqrt_38: "f32[512, 1, 1]", rsqrt_39: "f32[512, 1, 1]", convert_element_type_162: "bf16[512, 1, 1024]", amax_20: "f32[1, 16, 512, 1]", exp_20: "f32[1, 16, 512, 512]", rsqrt_40: "f32[512, 1, 1]", rsqrt_41: "f32[512, 1, 1]", convert_element_type_170: "bf16[512, 1, 1024]", amax_21: "f32[1, 16, 512, 1]", exp_21: "f32[1, 16, 512, 512]", rsqrt_42: "f32[512, 1, 1]", rsqrt_43: "f32[512, 1, 1]", convert_element_type_178: "bf16[512, 1, 1024]", amax_22: "f32[1, 16, 512, 1]", exp_22: "f32[1, 16, 512, 512]", rsqrt_44: "f32[512, 1, 1]", rsqrt_45: "f32[512, 1, 1]", convert_element_type_186: "bf16[512, 1, 1024]", amax_23: "f32[1, 16, 512, 1]", exp_23: "f32[1, 16, 512, 512]", rsqrt_46: "f32[512, 1, 1]", rsqrt_47: "f32[512, 1, 1]", addmm_48: "bf16[512, 32000]", arg1_1: "i64[1, 512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        slice_tensor: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(embedding, 0, -512, 9223372036854775807);  embedding = None
        detach_default: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax);  amax = None
        detach_default_1: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp);  exp = None
        detach_default_2: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt);  rsqrt = None
        detach_default_3: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_1);  rsqrt_1 = None
        slice_tensor_1: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_10, 0, -512, 9223372036854775807);  convert_element_type_10 = None
        detach_default_4: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_1);  amax_1 = None
        detach_default_5: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_1);  exp_1 = None
        detach_default_6: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_2);  rsqrt_2 = None
        detach_default_7: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_3);  rsqrt_3 = None
        slice_tensor_2: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_18, 0, -512, 9223372036854775807);  convert_element_type_18 = None
        detach_default_8: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_2);  amax_2 = None
        detach_default_9: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_2);  exp_2 = None
        detach_default_10: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_4);  rsqrt_4 = None
        detach_default_11: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_5);  rsqrt_5 = None
        slice_tensor_3: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_26, 0, -512, 9223372036854775807);  convert_element_type_26 = None
        detach_default_12: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_3);  amax_3 = None
        detach_default_13: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_3);  exp_3 = None
        detach_default_14: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_6);  rsqrt_6 = None
        detach_default_15: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_7);  rsqrt_7 = None
        slice_tensor_4: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_34, 0, -512, 9223372036854775807);  convert_element_type_34 = None
        detach_default_16: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_4);  amax_4 = None
        detach_default_17: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_4);  exp_4 = None
        detach_default_18: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_8);  rsqrt_8 = None
        detach_default_19: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_9);  rsqrt_9 = None
        slice_tensor_5: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_42, 0, -512, 9223372036854775807);  convert_element_type_42 = None
        detach_default_20: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_5);  amax_5 = None
        detach_default_21: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_5);  exp_5 = None
        detach_default_22: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_10);  rsqrt_10 = None
        detach_default_23: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_11);  rsqrt_11 = None
        slice_tensor_6: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_50, 0, -512, 9223372036854775807);  convert_element_type_50 = None
        detach_default_24: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_6);  amax_6 = None
        detach_default_25: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_6);  exp_6 = None
        detach_default_26: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_12);  rsqrt_12 = None
        detach_default_27: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_13);  rsqrt_13 = None
        slice_tensor_7: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_58, 0, -512, 9223372036854775807);  convert_element_type_58 = None
        detach_default_28: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_7);  amax_7 = None
        detach_default_29: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_7);  exp_7 = None
        detach_default_30: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_14);  rsqrt_14 = None
        detach_default_31: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_15);  rsqrt_15 = None
        slice_tensor_8: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_66, 0, -512, 9223372036854775807);  convert_element_type_66 = None
        detach_default_32: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_8);  amax_8 = None
        detach_default_33: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_8);  exp_8 = None
        detach_default_34: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_16);  rsqrt_16 = None
        detach_default_35: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_17);  rsqrt_17 = None
        slice_tensor_9: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_74, 0, -512, 9223372036854775807);  convert_element_type_74 = None
        detach_default_36: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_9);  amax_9 = None
        detach_default_37: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_9);  exp_9 = None
        detach_default_38: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_18);  rsqrt_18 = None
        detach_default_39: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_19);  rsqrt_19 = None
        slice_tensor_10: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_82, 0, -512, 9223372036854775807);  convert_element_type_82 = None
        detach_default_40: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_10);  amax_10 = None
        detach_default_41: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_10);  exp_10 = None
        detach_default_42: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_20);  rsqrt_20 = None
        detach_default_43: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_21);  rsqrt_21 = None
        slice_tensor_11: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_90, 0, -512, 9223372036854775807);  convert_element_type_90 = None
        detach_default_44: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_11);  amax_11 = None
        detach_default_45: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_11);  exp_11 = None
        detach_default_46: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_22);  rsqrt_22 = None
        detach_default_47: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_23);  rsqrt_23 = None
        slice_tensor_12: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_98, 0, -512, 9223372036854775807);  convert_element_type_98 = None
        detach_default_48: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_12);  amax_12 = None
        detach_default_49: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_12);  exp_12 = None
        detach_default_50: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_24);  rsqrt_24 = None
        detach_default_51: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_25);  rsqrt_25 = None
        slice_tensor_13: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_106, 0, -512, 9223372036854775807);  convert_element_type_106 = None
        detach_default_52: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_13);  amax_13 = None
        detach_default_53: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_13);  exp_13 = None
        detach_default_54: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_26);  rsqrt_26 = None
        detach_default_55: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_27);  rsqrt_27 = None
        slice_tensor_14: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_114, 0, -512, 9223372036854775807);  convert_element_type_114 = None
        detach_default_56: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_14);  amax_14 = None
        detach_default_57: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_14);  exp_14 = None
        detach_default_58: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_28);  rsqrt_28 = None
        detach_default_59: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_29);  rsqrt_29 = None
        slice_tensor_15: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_122, 0, -512, 9223372036854775807);  convert_element_type_122 = None
        detach_default_60: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_15);  amax_15 = None
        detach_default_61: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_15);  exp_15 = None
        detach_default_62: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_30);  rsqrt_30 = None
        detach_default_63: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_31);  rsqrt_31 = None
        slice_tensor_16: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_130, 0, -512, 9223372036854775807);  convert_element_type_130 = None
        detach_default_64: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_16);  amax_16 = None
        detach_default_65: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_16);  exp_16 = None
        detach_default_66: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_32);  rsqrt_32 = None
        detach_default_67: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_33);  rsqrt_33 = None
        slice_tensor_17: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_138, 0, -512, 9223372036854775807);  convert_element_type_138 = None
        detach_default_68: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_17);  amax_17 = None
        detach_default_69: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_17);  exp_17 = None
        detach_default_70: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_34);  rsqrt_34 = None
        detach_default_71: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_35);  rsqrt_35 = None
        slice_tensor_18: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_146, 0, -512, 9223372036854775807);  convert_element_type_146 = None
        detach_default_72: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_18);  amax_18 = None
        detach_default_73: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_18);  exp_18 = None
        detach_default_74: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_36);  rsqrt_36 = None
        detach_default_75: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_37);  rsqrt_37 = None
        slice_tensor_19: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_154, 0, -512, 9223372036854775807);  convert_element_type_154 = None
        detach_default_76: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_19);  amax_19 = None
        detach_default_77: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_19);  exp_19 = None
        detach_default_78: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_38);  rsqrt_38 = None
        detach_default_79: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_39);  rsqrt_39 = None
        slice_tensor_20: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_162, 0, -512, 9223372036854775807);  convert_element_type_162 = None
        detach_default_80: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_20);  amax_20 = None
        detach_default_81: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_20);  exp_20 = None
        detach_default_82: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_40);  rsqrt_40 = None
        detach_default_83: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_41);  rsqrt_41 = None
        slice_tensor_21: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_170, 0, -512, 9223372036854775807);  convert_element_type_170 = None
        detach_default_84: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_21);  amax_21 = None
        detach_default_85: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_21);  exp_21 = None
        detach_default_86: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_42);  rsqrt_42 = None
        detach_default_87: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_43);  rsqrt_43 = None
        slice_tensor_22: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_178, 0, -512, 9223372036854775807);  convert_element_type_178 = None
        detach_default_88: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_22);  amax_22 = None
        detach_default_89: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_22);  exp_22 = None
        detach_default_90: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_44);  rsqrt_44 = None
        detach_default_91: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_45);  rsqrt_45 = None
        slice_tensor_23: "bf16[512, 1, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_186, 0, -512, 9223372036854775807);  convert_element_type_186 = None
        detach_default_92: "f32[1, 16, 512, 1]" = torch.ops.aten.detach.default(amax_23);  amax_23 = None
        detach_default_93: "f32[1, 16, 512, 512]" = torch.ops.aten.detach.default(exp_23);  exp_23 = None
        detach_default_94: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_46);  rsqrt_46 = None
        detach_default_95: "f32[512, 1, 1]" = torch.ops.aten.detach.default(rsqrt_47);  rsqrt_47 = None
        view_default: "bf16[1, 512, 32000]" = torch.ops.aten.view.default(addmm_48, _shape_param_0);  addmm_48 = _shape_param_0 = None
        view_default_1: "bf16[512, 32000]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        view_default_2: "i64[512]" = torch.ops.aten.view.default(arg1_1, [-1]);  arg1_1 = None
        convert_element_type_default: "f32[512, 32000]" = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        amax_default: "f32[512, 1]" = torch.ops.aten.amax.default(convert_element_type_default, [1], True)
        detach_default_96: "f32[512, 1]" = torch.ops.aten.detach.default(amax_default)
        sub_tensor: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default);  convert_element_type_default = amax_default = None
        exp_default: "f32[512, 32000]" = torch.ops.aten.exp.default(sub_tensor)
        detach_default_97: "f32[512, 32000]" = torch.ops.aten.detach.default(exp_default)
        sum_dim_int_list: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True);  exp_default = None
        log_default: "f32[512, 1]" = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1: "f32[512, 32000]" = torch.ops.aten.sub.Tensor(sub_tensor, log_default);  sub_tensor = log_default = None
        convert_element_type_default_1: "bf16[512, 32000]" = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.bfloat16);  sub_tensor_1 = None
        ne_scalar: "b8[512]" = torch.ops.aten.ne.Scalar(view_default_2, -100)
        full_default: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[512]" = torch.ops.aten.where.self(ne_scalar, view_default_2, full_default);  view_default_2 = full_default = None
        unsqueeze_default: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default: "bf16[512, 1]" = torch.ops.aten.gather.default(convert_element_type_default_1, 1, unsqueeze_default);  convert_element_type_default_1 = unsqueeze_default = None
        squeeze_dim: "bf16[512]" = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default: "bf16[512]" = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        full_default_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "bf16[512]" = torch.ops.aten.where.self(ne_scalar, neg_default, full_default_1);  neg_default = full_default_1 = None
        sum_default: "i64[]" = torch.ops.aten.sum.default(ne_scalar);  ne_scalar = None
        convert_element_type_default_2: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_default, torch.bfloat16);  sum_default = None
        sum_default_1: "bf16[]" = torch.ops.aten.sum.default(where_self_1);  where_self_1 = None
        div_tensor: "bf16[]" = torch.ops.aten.div.Tensor(sum_default_1, convert_element_type_default_2);  sum_default_1 = convert_element_type_default_2 = None
        return (slice_tensor, detach_default, detach_default_1, detach_default_2, detach_default_3, slice_tensor_1, detach_default_4, detach_default_5, detach_default_6, detach_default_7, slice_tensor_2, detach_default_8, detach_default_9, detach_default_10, detach_default_11, slice_tensor_3, detach_default_12, detach_default_13, detach_default_14, detach_default_15, slice_tensor_4, detach_default_16, detach_default_17, detach_default_18, detach_default_19, slice_tensor_5, detach_default_20, detach_default_21, detach_default_22, detach_default_23, slice_tensor_6, detach_default_24, detach_default_25, detach_default_26, detach_default_27, slice_tensor_7, detach_default_28, detach_default_29, detach_default_30, detach_default_31, slice_tensor_8, detach_default_32, detach_default_33, detach_default_34, detach_default_35, slice_tensor_9, detach_default_36, detach_default_37, detach_default_38, detach_default_39, slice_tensor_10, detach_default_40, detach_default_41, detach_default_42, detach_default_43, slice_tensor_11, detach_default_44, detach_default_45, detach_default_46, detach_default_47, slice_tensor_12, detach_default_48, detach_default_49, detach_default_50, detach_default_51, slice_tensor_13, detach_default_52, detach_default_53, detach_default_54, detach_default_55, slice_tensor_14, detach_default_56, detach_default_57, detach_default_58, detach_default_59, slice_tensor_15, detach_default_60, detach_default_61, detach_default_62, detach_default_63, slice_tensor_16, detach_default_64, detach_default_65, detach_default_66, detach_default_67, slice_tensor_17, detach_default_68, detach_default_69, detach_default_70, detach_default_71, slice_tensor_18, detach_default_72, detach_default_73, detach_default_74, detach_default_75, slice_tensor_19, detach_default_76, detach_default_77, detach_default_78, detach_default_79, slice_tensor_20, detach_default_80, detach_default_81, detach_default_82, detach_default_83, slice_tensor_21, detach_default_84, detach_default_85, detach_default_86, detach_default_87, slice_tensor_22, detach_default_88, detach_default_89, detach_default_90, detach_default_91, slice_tensor_23, detach_default_92, detach_default_93, detach_default_94, detach_default_95, detach_default_96, detach_default_97, div_tensor)


def _default_make_inputs():
    return [
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 16, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16, 512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512, 32000], dtype=torch.bfloat16, device='cuda'),
    torch.randint(0, 512, [1, 512], dtype=torch.int64, device='cuda'),
    [1, 512, 32000],  # _shape_param_0
    [-1, 32000],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
