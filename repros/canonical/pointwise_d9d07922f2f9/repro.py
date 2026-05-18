"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g77
Pattern hash: d9d07922f2f9
Shape hash: be2f6cb4
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
    def forward(self, getitem_50: "f32[]", getitem_51: "f32[]", getitem_52: "f32[]", getitem_53: "f32[]", getitem_54: "f32[]", getitem_55: "f32[]", getitem_56: "f32[]", getitem_57: "f32[]", getitem_58: "f32[]", getitem_59: "f32[]", getitem_90: "f32[]", getitem_91: "f32[]", getitem_92: "f32[]", getitem_93: "f32[]", getitem_94: "f32[]", getitem_95: "f32[]", getitem_96: "f32[]", getitem_97: "f32[]", getitem_98: "f32[]", getitem_99: "f32[]", getitem_40: "f32[16, 1]", getitem_41: "f32[16]", getitem_42: "f32[16, 16]", getitem_43: "f32[16]", getitem_44: "f32[16, 16]", getitem_45: "f32[16]", getitem_46: "f32[16, 16]", getitem_47: "f32[16]", getitem_48: "f32[1, 16]", getitem_49: "f32[1]"):
        # No stacktrace found for following nodes
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59], 1);  getitem_50 = getitem_51 = getitem_52 = getitem_53 = getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = None
        getitem: "f32[]" = _foreach_sub_scalar[0]
        getitem_60: "f32[]" = _foreach_sub_scalar[1]
        getitem_61: "f32[]" = _foreach_sub_scalar[2]
        getitem_62: "f32[]" = _foreach_sub_scalar[3]
        getitem_63: "f32[]" = _foreach_sub_scalar[4]
        getitem_64: "f32[]" = _foreach_sub_scalar[5]
        getitem_65: "f32[]" = _foreach_sub_scalar[6]
        getitem_66: "f32[]" = _foreach_sub_scalar[7]
        getitem_67: "f32[]" = _foreach_sub_scalar[8]
        getitem_68: "f32[]" = _foreach_sub_scalar[9];  _foreach_sub_scalar = None
        _foreach_sqrt_default = torch.ops.aten._foreach_sqrt.default([getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99]);  getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = getitem_97 = getitem_98 = getitem_99 = None
        getitem_100: "f32[]" = _foreach_sqrt_default[0]
        getitem_101: "f32[]" = _foreach_sqrt_default[1]
        getitem_102: "f32[]" = _foreach_sqrt_default[2]
        getitem_103: "f32[]" = _foreach_sqrt_default[3]
        getitem_104: "f32[]" = _foreach_sqrt_default[4]
        getitem_105: "f32[]" = _foreach_sqrt_default[5]
        getitem_106: "f32[]" = _foreach_sqrt_default[6]
        getitem_107: "f32[]" = _foreach_sqrt_default[7]
        getitem_108: "f32[]" = _foreach_sqrt_default[8]
        getitem_109: "f32[]" = _foreach_sqrt_default[9];  _foreach_sqrt_default = None
        _foreach_sqrt_default_1 = torch.ops.aten._foreach_sqrt.default([getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49]);  getitem_40 = getitem_41 = getitem_42 = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = getitem_48 = getitem_49 = None
        getitem_69: "f32[16, 1]" = _foreach_sqrt_default_1[0]
        getitem_70: "f32[16]" = _foreach_sqrt_default_1[1]
        getitem_71: "f32[16, 16]" = _foreach_sqrt_default_1[2]
        getitem_72: "f32[16]" = _foreach_sqrt_default_1[3]
        getitem_73: "f32[16, 16]" = _foreach_sqrt_default_1[4]
        getitem_74: "f32[16]" = _foreach_sqrt_default_1[5]
        getitem_75: "f32[16, 16]" = _foreach_sqrt_default_1[6]
        getitem_76: "f32[16]" = _foreach_sqrt_default_1[7]
        getitem_77: "f32[1, 16]" = _foreach_sqrt_default_1[8]
        getitem_78: "f32[1]" = _foreach_sqrt_default_1[9];  _foreach_sqrt_default_1 = None
        return (getitem, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78)


def _default_make_inputs():
    return [
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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([16, 1], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([16], dtype=torch.float32, device='cuda'),
    torch.randn([1, 16], dtype=torch.float32, device='cuda'),
    torch.randn([1], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
