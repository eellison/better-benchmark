"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g77
Pattern hash: fb5ff4fa1bc8
Shape hash: 227d9b5e
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
    def forward(self, arg0_1: "f32[16, 1]", getitem_170: "f32[16, 1]", arg1_1: "f32[16]", getitem_171: "f32[16]", arg2_1: "f32[16, 16]", getitem_172: "f32[16, 16]", arg3_1: "f32[16]", getitem_173: "f32[16]", arg4_1: "f32[16, 16]", getitem_174: "f32[16, 16]", arg5_1: "f32[16]", getitem_175: "f32[16]", arg6_1: "f32[16, 16]", getitem_176: "f32[16, 16]", arg7_1: "f32[16]", getitem_177: "f32[16]", arg8_1: "f32[1, 16]", getitem_178: "f32[1, 16]", arg9_1: "f32[1]", getitem_179: "f32[1]", arg10_1: "f32[]", getitem_1: "f32[]", arg11_1: "f32[16]", getitem_21: "f32[16]", arg12_1: "f32[16]", getitem_41: "f32[16]", arg13_1: "f32[16, 1]", getitem_20: "f32[16, 1]", arg14_1: "f32[16, 1]", getitem_40: "f32[16, 1]", arg15_1: "f32[]", getitem: "f32[]", arg16_1: "f32[]", getitem_2: "f32[]", arg17_1: "f32[]", getitem_3: "f32[]", arg18_1: "f32[]", getitem_4: "f32[]", arg19_1: "f32[]", getitem_5: "f32[]", arg20_1: "f32[]", getitem_6: "f32[]", arg21_1: "f32[]", getitem_7: "f32[]", arg22_1: "f32[]", getitem_8: "f32[]", arg23_1: "f32[]", getitem_9: "f32[]", arg24_1: "f32[16, 16]", getitem_22: "f32[16, 16]", arg25_1: "f32[16]", getitem_23: "f32[16]", arg26_1: "f32[16, 16]", getitem_24: "f32[16, 16]", arg27_1: "f32[16]", getitem_25: "f32[16]", arg28_1: "f32[16, 16]", getitem_26: "f32[16, 16]", arg29_1: "f32[16]", getitem_27: "f32[16]", arg30_1: "f32[1, 16]", getitem_28: "f32[1, 16]", arg31_1: "f32[1]", getitem_29: "f32[1]", arg32_1: "f32[16, 16]", getitem_42: "f32[16, 16]", arg33_1: "f32[16]", getitem_43: "f32[16]", arg34_1: "f32[16, 16]", getitem_44: "f32[16, 16]", arg35_1: "f32[16]", getitem_45: "f32[16]", arg36_1: "f32[16, 16]", getitem_46: "f32[16, 16]", arg37_1: "f32[16]", getitem_47: "f32[16]", arg38_1: "f32[1, 16]", getitem_48: "f32[1, 16]", arg39_1: "f32[1]", getitem_49: "f32[1]"):
        # No stacktrace found for following nodes
        copy__default: "f32[16, 1]" = torch.ops.aten.copy_.default(arg0_1, getitem_170);  arg0_1 = getitem_170 = None
        copy__default_1: "f32[16]" = torch.ops.aten.copy_.default(arg1_1, getitem_171);  arg1_1 = getitem_171 = None
        copy__default_2: "f32[16, 16]" = torch.ops.aten.copy_.default(arg2_1, getitem_172);  arg2_1 = getitem_172 = None
        copy__default_3: "f32[16]" = torch.ops.aten.copy_.default(arg3_1, getitem_173);  arg3_1 = getitem_173 = None
        copy__default_4: "f32[16, 16]" = torch.ops.aten.copy_.default(arg4_1, getitem_174);  arg4_1 = getitem_174 = None
        copy__default_5: "f32[16]" = torch.ops.aten.copy_.default(arg5_1, getitem_175);  arg5_1 = getitem_175 = None
        copy__default_6: "f32[16, 16]" = torch.ops.aten.copy_.default(arg6_1, getitem_176);  arg6_1 = getitem_176 = None
        copy__default_7: "f32[16]" = torch.ops.aten.copy_.default(arg7_1, getitem_177);  arg7_1 = getitem_177 = None
        copy__default_8: "f32[1, 16]" = torch.ops.aten.copy_.default(arg8_1, getitem_178);  arg8_1 = getitem_178 = None
        copy__default_9: "f32[1]" = torch.ops.aten.copy_.default(arg9_1, getitem_179);  arg9_1 = getitem_179 = None
        copy__default_10: "f32[]" = torch.ops.aten.copy_.default(arg10_1, getitem_1);  arg10_1 = getitem_1 = None
        copy__default_11: "f32[16]" = torch.ops.aten.copy_.default(arg11_1, getitem_21);  arg11_1 = getitem_21 = None
        copy__default_12: "f32[16]" = torch.ops.aten.copy_.default(arg12_1, getitem_41);  arg12_1 = getitem_41 = None
        copy__default_13: "f32[16, 1]" = torch.ops.aten.copy_.default(arg13_1, getitem_20);  arg13_1 = getitem_20 = None
        copy__default_14: "f32[16, 1]" = torch.ops.aten.copy_.default(arg14_1, getitem_40);  arg14_1 = getitem_40 = None
        copy__default_15: "f32[]" = torch.ops.aten.copy_.default(arg15_1, getitem);  arg15_1 = getitem = None
        copy__default_16: "f32[]" = torch.ops.aten.copy_.default(arg16_1, getitem_2);  arg16_1 = getitem_2 = None
        copy__default_17: "f32[]" = torch.ops.aten.copy_.default(arg17_1, getitem_3);  arg17_1 = getitem_3 = None
        copy__default_18: "f32[]" = torch.ops.aten.copy_.default(arg18_1, getitem_4);  arg18_1 = getitem_4 = None
        copy__default_19: "f32[]" = torch.ops.aten.copy_.default(arg19_1, getitem_5);  arg19_1 = getitem_5 = None
        copy__default_20: "f32[]" = torch.ops.aten.copy_.default(arg20_1, getitem_6);  arg20_1 = getitem_6 = None
        copy__default_21: "f32[]" = torch.ops.aten.copy_.default(arg21_1, getitem_7);  arg21_1 = getitem_7 = None
        copy__default_22: "f32[]" = torch.ops.aten.copy_.default(arg22_1, getitem_8);  arg22_1 = getitem_8 = None
        copy__default_23: "f32[]" = torch.ops.aten.copy_.default(arg23_1, getitem_9);  arg23_1 = getitem_9 = None
        copy__default_24: "f32[16, 16]" = torch.ops.aten.copy_.default(arg24_1, getitem_22);  arg24_1 = getitem_22 = None
        copy__default_25: "f32[16]" = torch.ops.aten.copy_.default(arg25_1, getitem_23);  arg25_1 = getitem_23 = None
        copy__default_26: "f32[16, 16]" = torch.ops.aten.copy_.default(arg26_1, getitem_24);  arg26_1 = getitem_24 = None
        copy__default_27: "f32[16]" = torch.ops.aten.copy_.default(arg27_1, getitem_25);  arg27_1 = getitem_25 = None
        copy__default_28: "f32[16, 16]" = torch.ops.aten.copy_.default(arg28_1, getitem_26);  arg28_1 = getitem_26 = None
        copy__default_29: "f32[16]" = torch.ops.aten.copy_.default(arg29_1, getitem_27);  arg29_1 = getitem_27 = None
        copy__default_30: "f32[1, 16]" = torch.ops.aten.copy_.default(arg30_1, getitem_28);  arg30_1 = getitem_28 = None
        copy__default_31: "f32[1]" = torch.ops.aten.copy_.default(arg31_1, getitem_29);  arg31_1 = getitem_29 = None
        copy__default_32: "f32[16, 16]" = torch.ops.aten.copy_.default(arg32_1, getitem_42);  arg32_1 = getitem_42 = None
        copy__default_33: "f32[16]" = torch.ops.aten.copy_.default(arg33_1, getitem_43);  arg33_1 = getitem_43 = None
        copy__default_34: "f32[16, 16]" = torch.ops.aten.copy_.default(arg34_1, getitem_44);  arg34_1 = getitem_44 = None
        copy__default_35: "f32[16]" = torch.ops.aten.copy_.default(arg35_1, getitem_45);  arg35_1 = getitem_45 = None
        copy__default_36: "f32[16, 16]" = torch.ops.aten.copy_.default(arg36_1, getitem_46);  arg36_1 = getitem_46 = None
        copy__default_37: "f32[16]" = torch.ops.aten.copy_.default(arg37_1, getitem_47);  arg37_1 = getitem_47 = None
        copy__default_38: "f32[1, 16]" = torch.ops.aten.copy_.default(arg38_1, getitem_48);  arg38_1 = getitem_48 = None
        copy__default_39: "f32[1]" = torch.ops.aten.copy_.default(arg39_1, getitem_49);  arg39_1 = getitem_49 = None
        return (copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39)


def _default_make_inputs():
    return [
    torch.randn([16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn([1], dtype=torch.float32, device='cuda'),
    torch.randn([1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn([1], dtype=torch.float32, device='cuda'),
    torch.randn([1], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn([1], dtype=torch.float32, device='cuda'),
    torch.randn([1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
