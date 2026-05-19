"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-6-6-linux.aws.a100_graph32
Pattern hash: b8df75a78700
Shape hash: 091914fd
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
    def forward(self, arg11_1: "f32[]", arg6_1: "f32[]", arg12_1: "f32[]", arg13_1: "f32[]", arg14_1: "f32[]", arg15_1: "f32[]", arg24_1: "f32[1024, 3]", arg25_1: "f32[1024]", arg26_1: "f32[1024, 1024]", arg27_1: "f32[1024]", arg28_1: "f32[2, 1024]", arg29_1: "f32[2]", arg9_1: "f32[1024, 3]", arg7_1: "f32[1024]", arg16_1: "f32[1024, 1024]", arg17_1: "f32[1024]", arg18_1: "f32[2, 1024]", arg19_1: "f32[2]", arg10_1: "f32[1024, 3]", arg8_1: "f32[1024]", arg20_1: "f32[1024, 1024]", arg21_1: "f32[1024]", arg22_1: "f32[2, 1024]", arg23_1: "f32[2]", arg0_1: "f32[1024, 3]", arg1_1: "f32[1024]", arg2_1: "f32[1024, 1024]", arg3_1: "f32[1024]", arg4_1: "f32[2, 1024]", arg5_1: "f32[2]"):
        # No stacktrace found for following nodes
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([arg11_1, arg6_1, arg12_1, arg13_1, arg14_1, arg15_1], 1)
        getitem: "f32[]" = _foreach_add_scalar[0]
        getitem_1: "f32[]" = _foreach_add_scalar[1]
        getitem_2: "f32[]" = _foreach_add_scalar[2]
        getitem_3: "f32[]" = _foreach_add_scalar[3]
        getitem_4: "f32[]" = _foreach_add_scalar[4]
        getitem_5: "f32[]" = _foreach_add_scalar[5];  _foreach_add_scalar = None
        _foreach_sub_list = torch.ops.aten._foreach_sub.List([arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1], [arg9_1, arg7_1, arg16_1, arg17_1, arg18_1, arg19_1])
        getitem_6: "f32[1024, 3]" = _foreach_sub_list[0]
        getitem_7: "f32[1024]" = _foreach_sub_list[1]
        getitem_8: "f32[1024, 1024]" = _foreach_sub_list[2]
        getitem_9: "f32[1024]" = _foreach_sub_list[3]
        getitem_10: "f32[2, 1024]" = _foreach_sub_list[4]
        getitem_11: "f32[2]" = _foreach_sub_list[5];  _foreach_sub_list = None
        full_default: "f32[1024, 3]" = torch.ops.aten.full.default([1024, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[1024]" = torch.ops.aten.full.default([1024], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[1024, 1024]" = torch.ops.aten.full.default([1024, 1024], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[1024]" = torch.ops.aten.full.default([1024], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[2, 1024]" = torch.ops.aten.full.default([2, 1024], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[2]" = torch.ops.aten.full.default([2], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([arg9_1, arg7_1, arg16_1, arg17_1, arg18_1, arg19_1], [full_default, full_default_1, full_default_2, full_default_3, full_default_4, full_default_5], [getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11]);  full_default = full_default_1 = full_default_2 = full_default_3 = full_default_4 = full_default_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = None
        getitem_12: "f32[1024, 3]" = _foreach_addcmul_scalar[0]
        getitem_13: "f32[1024]" = _foreach_addcmul_scalar[1]
        getitem_14: "f32[1024, 1024]" = _foreach_addcmul_scalar[2]
        getitem_15: "f32[1024]" = _foreach_addcmul_scalar[3]
        getitem_16: "f32[2, 1024]" = _foreach_addcmul_scalar[4]
        getitem_17: "f32[2]" = _foreach_addcmul_scalar[5];  _foreach_addcmul_scalar = None
        _foreach_mul_scalar = torch.ops.aten._foreach_mul.Scalar([arg10_1, arg8_1, arg20_1, arg21_1, arg22_1, arg23_1], 0.999)
        getitem_18: "f32[1024, 3]" = _foreach_mul_scalar[0]
        getitem_19: "f32[1024]" = _foreach_mul_scalar[1]
        getitem_20: "f32[1024, 1024]" = _foreach_mul_scalar[2]
        getitem_21: "f32[1024]" = _foreach_mul_scalar[3]
        getitem_22: "f32[2, 1024]" = _foreach_mul_scalar[4]
        getitem_23: "f32[2]" = _foreach_mul_scalar[5];  _foreach_mul_scalar = None
        _foreach_addcmul_scalar_1 = torch.ops.aten._foreach_addcmul.Scalar([getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23], [arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1], [arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1], 0.0010000000000000009);  getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = None
        getitem_24: "f32[1024, 3]" = _foreach_addcmul_scalar_1[0]
        getitem_25: "f32[1024]" = _foreach_addcmul_scalar_1[1]
        getitem_26: "f32[1024, 1024]" = _foreach_addcmul_scalar_1[2]
        getitem_27: "f32[1024]" = _foreach_addcmul_scalar_1[3]
        getitem_28: "f32[2, 1024]" = _foreach_addcmul_scalar_1[4]
        getitem_29: "f32[2]" = _foreach_addcmul_scalar_1[5];  _foreach_addcmul_scalar_1 = None
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5])
        getitem_30: "f32[]" = _foreach_pow_scalar_and_tensor[0]
        getitem_31: "f32[]" = _foreach_pow_scalar_and_tensor[1]
        getitem_32: "f32[]" = _foreach_pow_scalar_and_tensor[2]
        getitem_33: "f32[]" = _foreach_pow_scalar_and_tensor[3]
        getitem_34: "f32[]" = _foreach_pow_scalar_and_tensor[4]
        getitem_35: "f32[]" = _foreach_pow_scalar_and_tensor[5];  _foreach_pow_scalar_and_tensor = None
        _foreach_pow_scalar_and_tensor_1 = torch.ops.aten._foreach_pow.ScalarAndTensor(0.999, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5])
        getitem_36: "f32[]" = _foreach_pow_scalar_and_tensor_1[0]
        getitem_37: "f32[]" = _foreach_pow_scalar_and_tensor_1[1]
        getitem_38: "f32[]" = _foreach_pow_scalar_and_tensor_1[2]
        getitem_39: "f32[]" = _foreach_pow_scalar_and_tensor_1[3]
        getitem_40: "f32[]" = _foreach_pow_scalar_and_tensor_1[4]
        getitem_41: "f32[]" = _foreach_pow_scalar_and_tensor_1[5];  _foreach_pow_scalar_and_tensor_1 = None
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35], 1);  getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = None
        getitem_42: "f32[]" = _foreach_sub_scalar[0]
        getitem_43: "f32[]" = _foreach_sub_scalar[1]
        getitem_44: "f32[]" = _foreach_sub_scalar[2]
        getitem_45: "f32[]" = _foreach_sub_scalar[3]
        getitem_46: "f32[]" = _foreach_sub_scalar[4]
        getitem_47: "f32[]" = _foreach_sub_scalar[5];  _foreach_sub_scalar = None
        _foreach_sub_scalar_1 = torch.ops.aten._foreach_sub.Scalar([getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41], 1);  getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = None
        getitem_48: "f32[]" = _foreach_sub_scalar_1[0]
        getitem_49: "f32[]" = _foreach_sub_scalar_1[1]
        getitem_50: "f32[]" = _foreach_sub_scalar_1[2]
        getitem_51: "f32[]" = _foreach_sub_scalar_1[3]
        getitem_52: "f32[]" = _foreach_sub_scalar_1[4]
        getitem_53: "f32[]" = _foreach_sub_scalar_1[5];  _foreach_sub_scalar_1 = None
        _foreach_neg_default = torch.ops.aten._foreach_neg.default([getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53]);  getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = None
        getitem_54: "f32[]" = _foreach_neg_default[0]
        getitem_55: "f32[]" = _foreach_neg_default[1]
        getitem_56: "f32[]" = _foreach_neg_default[2]
        getitem_57: "f32[]" = _foreach_neg_default[3]
        getitem_58: "f32[]" = _foreach_neg_default[4]
        getitem_59: "f32[]" = _foreach_neg_default[5];  _foreach_neg_default = None
        _foreach_div_scalar = torch.ops.aten._foreach_div.Scalar([getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47], 0.01);  getitem_42 = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = None
        getitem_60: "f32[]" = _foreach_div_scalar[0]
        getitem_61: "f32[]" = _foreach_div_scalar[1]
        getitem_62: "f32[]" = _foreach_div_scalar[2]
        getitem_63: "f32[]" = _foreach_div_scalar[3]
        getitem_64: "f32[]" = _foreach_div_scalar[4]
        getitem_65: "f32[]" = _foreach_div_scalar[5];  _foreach_div_scalar = None
        _foreach_reciprocal_default = torch.ops.aten._foreach_reciprocal.default([getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65]);  getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = None
        getitem_66: "f32[]" = _foreach_reciprocal_default[0]
        getitem_67: "f32[]" = _foreach_reciprocal_default[1]
        getitem_68: "f32[]" = _foreach_reciprocal_default[2]
        getitem_69: "f32[]" = _foreach_reciprocal_default[3]
        getitem_70: "f32[]" = _foreach_reciprocal_default[4]
        getitem_71: "f32[]" = _foreach_reciprocal_default[5];  _foreach_reciprocal_default = None
        _foreach_sqrt_default = torch.ops.aten._foreach_sqrt.default([getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59]);  getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = None
        getitem_72: "f32[]" = _foreach_sqrt_default[0]
        getitem_73: "f32[]" = _foreach_sqrt_default[1]
        getitem_74: "f32[]" = _foreach_sqrt_default[2]
        getitem_75: "f32[]" = _foreach_sqrt_default[3]
        getitem_76: "f32[]" = _foreach_sqrt_default[4]
        getitem_77: "f32[]" = _foreach_sqrt_default[5];  _foreach_sqrt_default = None
        _foreach_sqrt_default_1 = torch.ops.aten._foreach_sqrt.default([getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29])
        getitem_78: "f32[1024, 3]" = _foreach_sqrt_default_1[0]
        getitem_79: "f32[1024]" = _foreach_sqrt_default_1[1]
        getitem_80: "f32[1024, 1024]" = _foreach_sqrt_default_1[2]
        getitem_81: "f32[1024]" = _foreach_sqrt_default_1[3]
        getitem_82: "f32[2, 1024]" = _foreach_sqrt_default_1[4]
        getitem_83: "f32[2]" = _foreach_sqrt_default_1[5];  _foreach_sqrt_default_1 = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83], [getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77]);  getitem_78 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = None
        getitem_84: "f32[1024, 3]" = _foreach_div_list[0]
        getitem_85: "f32[1024]" = _foreach_div_list[1]
        getitem_86: "f32[1024, 1024]" = _foreach_div_list[2]
        getitem_87: "f32[1024]" = _foreach_div_list[3]
        getitem_88: "f32[2, 1024]" = _foreach_div_list[4]
        getitem_89: "f32[2]" = _foreach_div_list[5];  _foreach_div_list = None
        _foreach_add_scalar_1 = torch.ops.aten._foreach_add.Scalar([getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89], 1e-08);  getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = None
        getitem_90: "f32[1024, 3]" = _foreach_add_scalar_1[0]
        getitem_91: "f32[1024]" = _foreach_add_scalar_1[1]
        getitem_92: "f32[1024, 1024]" = _foreach_add_scalar_1[2]
        getitem_93: "f32[1024]" = _foreach_add_scalar_1[3]
        getitem_94: "f32[2, 1024]" = _foreach_add_scalar_1[4]
        getitem_95: "f32[2]" = _foreach_add_scalar_1[5];  _foreach_add_scalar_1 = None
        _foreach_div_list_1 = torch.ops.aten._foreach_div.List([getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95], [getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71]);  getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = None
        getitem_96: "f32[1024, 3]" = _foreach_div_list_1[0]
        getitem_97: "f32[1024]" = _foreach_div_list_1[1]
        getitem_98: "f32[1024, 1024]" = _foreach_div_list_1[2]
        getitem_99: "f32[1024]" = _foreach_div_list_1[3]
        getitem_100: "f32[2, 1024]" = _foreach_div_list_1[4]
        getitem_101: "f32[2]" = _foreach_div_list_1[5];  _foreach_div_list_1 = None
        _foreach_addcdiv_scalar = torch.ops.aten._foreach_addcdiv.Scalar([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1], [getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17], [getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101]);  getitem_96 = getitem_97 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = None
        getitem_102: "f32[1024, 3]" = _foreach_addcdiv_scalar[0]
        getitem_103: "f32[1024]" = _foreach_addcdiv_scalar[1]
        getitem_104: "f32[1024, 1024]" = _foreach_addcdiv_scalar[2]
        getitem_105: "f32[1024]" = _foreach_addcdiv_scalar[3]
        getitem_106: "f32[2, 1024]" = _foreach_addcdiv_scalar[4]
        getitem_107: "f32[2]" = _foreach_addcdiv_scalar[5];  _foreach_addcdiv_scalar = None
        copy__default: "f32[1024, 3]" = torch.ops.aten.copy_.default(arg0_1, getitem_102);  arg0_1 = getitem_102 = None
        copy__default_1: "f32[1024]" = torch.ops.aten.copy_.default(arg1_1, getitem_103);  arg1_1 = getitem_103 = None
        copy__default_2: "f32[1024, 1024]" = torch.ops.aten.copy_.default(arg2_1, getitem_104);  arg2_1 = getitem_104 = None
        copy__default_3: "f32[1024]" = torch.ops.aten.copy_.default(arg3_1, getitem_105);  arg3_1 = getitem_105 = None
        copy__default_4: "f32[2, 1024]" = torch.ops.aten.copy_.default(arg4_1, getitem_106);  arg4_1 = getitem_106 = None
        copy__default_5: "f32[2]" = torch.ops.aten.copy_.default(arg5_1, getitem_107);  arg5_1 = getitem_107 = None
        copy__default_6: "f32[]" = torch.ops.aten.copy_.default(arg6_1, getitem_1);  arg6_1 = getitem_1 = None
        copy__default_7: "f32[1024]" = torch.ops.aten.copy_.default(arg7_1, getitem_13);  arg7_1 = getitem_13 = None
        copy__default_8: "f32[1024]" = torch.ops.aten.copy_.default(arg8_1, getitem_25);  arg8_1 = getitem_25 = None
        copy__default_9: "f32[1024, 3]" = torch.ops.aten.copy_.default(arg9_1, getitem_12);  arg9_1 = getitem_12 = None
        copy__default_10: "f32[1024, 3]" = torch.ops.aten.copy_.default(arg10_1, getitem_24);  arg10_1 = getitem_24 = None
        copy__default_11: "f32[]" = torch.ops.aten.copy_.default(arg11_1, getitem);  arg11_1 = getitem = None
        copy__default_12: "f32[]" = torch.ops.aten.copy_.default(arg12_1, getitem_2);  arg12_1 = getitem_2 = None
        copy__default_13: "f32[]" = torch.ops.aten.copy_.default(arg13_1, getitem_3);  arg13_1 = getitem_3 = None
        copy__default_14: "f32[]" = torch.ops.aten.copy_.default(arg14_1, getitem_4);  arg14_1 = getitem_4 = None
        copy__default_15: "f32[]" = torch.ops.aten.copy_.default(arg15_1, getitem_5);  arg15_1 = getitem_5 = None
        copy__default_16: "f32[1024, 1024]" = torch.ops.aten.copy_.default(arg16_1, getitem_14);  arg16_1 = getitem_14 = None
        copy__default_17: "f32[1024]" = torch.ops.aten.copy_.default(arg17_1, getitem_15);  arg17_1 = getitem_15 = None
        copy__default_18: "f32[2, 1024]" = torch.ops.aten.copy_.default(arg18_1, getitem_16);  arg18_1 = getitem_16 = None
        copy__default_19: "f32[2]" = torch.ops.aten.copy_.default(arg19_1, getitem_17);  arg19_1 = getitem_17 = None
        copy__default_20: "f32[1024, 1024]" = torch.ops.aten.copy_.default(arg20_1, getitem_26);  arg20_1 = getitem_26 = None
        copy__default_21: "f32[1024]" = torch.ops.aten.copy_.default(arg21_1, getitem_27);  arg21_1 = getitem_27 = None
        copy__default_22: "f32[2, 1024]" = torch.ops.aten.copy_.default(arg22_1, getitem_28);  arg22_1 = getitem_28 = None
        copy__default_23: "f32[2]" = torch.ops.aten.copy_.default(arg23_1, getitem_29);  arg23_1 = getitem_29 = None
        return (copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23)


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 3], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([2, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 3], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([2, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 3], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([2, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 3], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([2, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([2], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
