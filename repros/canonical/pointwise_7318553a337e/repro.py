"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g244
Pattern hash: 7318553a337e
Shape hash: 84120312
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg24_1: "f32[64]", arg25_1: "f32[64, 64]", arg26_1: "f32[64]", arg27_1: "f32[64, 64]", arg28_1: "f32[64]", arg29_1: "f32[64, 64]", arg9_1: "f32[64]", arg7_1: "f32[64, 64]", arg16_1: "f32[64]", arg17_1: "f32[64, 64]", arg18_1: "f32[64]", arg19_1: "f32[64, 64]", getitem_60: "f32[]", getitem_61: "f32[]", getitem_62: "f32[]", getitem_63: "f32[]", getitem_64: "f32[]", getitem_65: "f32[]", getitem_84: "f32[64]", getitem_85: "f32[64, 64]", getitem_86: "f32[64]", getitem_87: "f32[64, 64]", getitem_88: "f32[64]", getitem_89: "f32[64, 64]"):
        # No stacktrace found for following nodes
        _foreach_sub_list = torch.ops.aten._foreach_sub.List([arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1], [arg9_1, arg7_1, arg16_1, arg17_1, arg18_1, arg19_1]);  arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = arg9_1 = arg7_1 = arg16_1 = arg17_1 = arg18_1 = arg19_1 = None
        getitem: "f32[64]" = _foreach_sub_list[0]
        getitem_1: "f32[64, 64]" = _foreach_sub_list[1]
        getitem_2: "f32[64]" = _foreach_sub_list[2]
        getitem_3: "f32[64, 64]" = _foreach_sub_list[3]
        getitem_4: "f32[64]" = _foreach_sub_list[4]
        getitem_5: "f32[64, 64]" = _foreach_sub_list[5];  _foreach_sub_list = None
        _foreach_reciprocal_default = torch.ops.aten._foreach_reciprocal.default([getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65]);  getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = None
        getitem_66: "f32[]" = _foreach_reciprocal_default[0]
        getitem_67: "f32[]" = _foreach_reciprocal_default[1]
        getitem_68: "f32[]" = _foreach_reciprocal_default[2]
        getitem_69: "f32[]" = _foreach_reciprocal_default[3]
        getitem_70: "f32[]" = _foreach_reciprocal_default[4]
        getitem_71: "f32[]" = _foreach_reciprocal_default[5];  _foreach_reciprocal_default = None
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89], 1e-08);  getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = None
        getitem_90: "f32[64]" = _foreach_add_scalar[0]
        getitem_91: "f32[64, 64]" = _foreach_add_scalar[1]
        getitem_92: "f32[64]" = _foreach_add_scalar[2]
        getitem_93: "f32[64, 64]" = _foreach_add_scalar[3]
        getitem_94: "f32[64]" = _foreach_add_scalar[4]
        getitem_95: "f32[64, 64]" = _foreach_add_scalar[5];  _foreach_add_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95)


def _default_make_inputs():
    return [
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 64], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
