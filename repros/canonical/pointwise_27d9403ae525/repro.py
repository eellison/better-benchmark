"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g244
Pattern hash: 27d9403ae525
Shape hash: cc111085
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[64]", getitem_102: "f32[64]", arg1_1: "f32[64, 64]", getitem_103: "f32[64, 64]", arg2_1: "f32[64]", getitem_104: "f32[64]", arg3_1: "f32[64, 64]", getitem_105: "f32[64, 64]", arg4_1: "f32[64]", getitem_106: "f32[64]", arg5_1: "f32[64, 64]", getitem_107: "f32[64, 64]", arg6_1: "f32[]", getitem_1: "f32[]", arg7_1: "f32[64, 64]", getitem_13: "f32[64, 64]", arg8_1: "f32[64, 64]", getitem_25: "f32[64, 64]", arg9_1: "f32[64]", getitem_12: "f32[64]", arg10_1: "f32[64]", getitem_24: "f32[64]", arg11_1: "f32[]", getitem: "f32[]", arg12_1: "f32[]", getitem_2: "f32[]", arg13_1: "f32[]", getitem_3: "f32[]", arg14_1: "f32[]", getitem_4: "f32[]", arg15_1: "f32[]", getitem_5: "f32[]", arg16_1: "f32[64]", getitem_14: "f32[64]", arg17_1: "f32[64, 64]", getitem_15: "f32[64, 64]", arg18_1: "f32[64]", getitem_16: "f32[64]", arg19_1: "f32[64, 64]", getitem_17: "f32[64, 64]", arg20_1: "f32[64]", getitem_26: "f32[64]", arg21_1: "f32[64, 64]", getitem_27: "f32[64, 64]", arg22_1: "f32[64]", getitem_28: "f32[64]", arg23_1: "f32[64, 64]", getitem_29: "f32[64, 64]"):
        # No stacktrace found for following nodes
        copy__default: "f32[64]" = torch.ops.aten.copy_.default(arg0_1, getitem_102);  arg0_1 = getitem_102 = None
        copy__default_1: "f32[64, 64]" = torch.ops.aten.copy_.default(arg1_1, getitem_103);  arg1_1 = getitem_103 = None
        copy__default_2: "f32[64]" = torch.ops.aten.copy_.default(arg2_1, getitem_104);  arg2_1 = getitem_104 = None
        copy__default_3: "f32[64, 64]" = torch.ops.aten.copy_.default(arg3_1, getitem_105);  arg3_1 = getitem_105 = None
        copy__default_4: "f32[64]" = torch.ops.aten.copy_.default(arg4_1, getitem_106);  arg4_1 = getitem_106 = None
        copy__default_5: "f32[64, 64]" = torch.ops.aten.copy_.default(arg5_1, getitem_107);  arg5_1 = getitem_107 = None
        copy__default_6: "f32[]" = torch.ops.aten.copy_.default(arg6_1, getitem_1);  arg6_1 = getitem_1 = None
        copy__default_7: "f32[64, 64]" = torch.ops.aten.copy_.default(arg7_1, getitem_13);  arg7_1 = getitem_13 = None
        copy__default_8: "f32[64, 64]" = torch.ops.aten.copy_.default(arg8_1, getitem_25);  arg8_1 = getitem_25 = None
        copy__default_9: "f32[64]" = torch.ops.aten.copy_.default(arg9_1, getitem_12);  arg9_1 = getitem_12 = None
        copy__default_10: "f32[64]" = torch.ops.aten.copy_.default(arg10_1, getitem_24);  arg10_1 = getitem_24 = None
        copy__default_11: "f32[]" = torch.ops.aten.copy_.default(arg11_1, getitem);  arg11_1 = getitem = None
        copy__default_12: "f32[]" = torch.ops.aten.copy_.default(arg12_1, getitem_2);  arg12_1 = getitem_2 = None
        copy__default_13: "f32[]" = torch.ops.aten.copy_.default(arg13_1, getitem_3);  arg13_1 = getitem_3 = None
        copy__default_14: "f32[]" = torch.ops.aten.copy_.default(arg14_1, getitem_4);  arg14_1 = getitem_4 = None
        copy__default_15: "f32[]" = torch.ops.aten.copy_.default(arg15_1, getitem_5);  arg15_1 = getitem_5 = None
        copy__default_16: "f32[64]" = torch.ops.aten.copy_.default(arg16_1, getitem_14);  arg16_1 = getitem_14 = None
        copy__default_17: "f32[64, 64]" = torch.ops.aten.copy_.default(arg17_1, getitem_15);  arg17_1 = getitem_15 = None
        copy__default_18: "f32[64]" = torch.ops.aten.copy_.default(arg18_1, getitem_16);  arg18_1 = getitem_16 = None
        copy__default_19: "f32[64, 64]" = torch.ops.aten.copy_.default(arg19_1, getitem_17);  arg19_1 = getitem_17 = None
        copy__default_20: "f32[64]" = torch.ops.aten.copy_.default(arg20_1, getitem_26);  arg20_1 = getitem_26 = None
        copy__default_21: "f32[64, 64]" = torch.ops.aten.copy_.default(arg21_1, getitem_27);  arg21_1 = getitem_27 = None
        copy__default_22: "f32[64]" = torch.ops.aten.copy_.default(arg22_1, getitem_28);  arg22_1 = getitem_28 = None
        copy__default_23: "f32[64, 64]" = torch.ops.aten.copy_.default(arg23_1, getitem_29);  arg23_1 = getitem_29 = None
        return (copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23)


def _default_make_inputs():
    return [
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
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
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
