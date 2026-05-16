"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g244
Pattern hash: 7c9b0fdf5a91
Shape hash: 84c9dc21
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_30: "f32[]", getitem_31: "f32[]", getitem_32: "f32[]", getitem_33: "f32[]", getitem_34: "f32[]", getitem_35: "f32[]", getitem_54: "f32[]", getitem_55: "f32[]", getitem_56: "f32[]", getitem_57: "f32[]", getitem_58: "f32[]", getitem_59: "f32[]", getitem_24: "f32[64]", getitem_25: "f32[64, 64]", getitem_26: "f32[64]", getitem_27: "f32[64, 64]", getitem_28: "f32[64]", getitem_29: "f32[64, 64]"):
        # No stacktrace found for following nodes
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35], 1);  getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = None
        getitem: "f32[]" = _foreach_sub_scalar[0]
        getitem_36: "f32[]" = _foreach_sub_scalar[1]
        getitem_37: "f32[]" = _foreach_sub_scalar[2]
        getitem_38: "f32[]" = _foreach_sub_scalar[3]
        getitem_39: "f32[]" = _foreach_sub_scalar[4]
        getitem_40: "f32[]" = _foreach_sub_scalar[5];  _foreach_sub_scalar = None
        _foreach_sqrt_default = torch.ops.aten._foreach_sqrt.default([getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59]);  getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = None
        getitem_60: "f32[]" = _foreach_sqrt_default[0]
        getitem_61: "f32[]" = _foreach_sqrt_default[1]
        getitem_62: "f32[]" = _foreach_sqrt_default[2]
        getitem_63: "f32[]" = _foreach_sqrt_default[3]
        getitem_64: "f32[]" = _foreach_sqrt_default[4]
        getitem_65: "f32[]" = _foreach_sqrt_default[5];  _foreach_sqrt_default = None
        _foreach_sqrt_default_1 = torch.ops.aten._foreach_sqrt.default([getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29]);  getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = None
        getitem_41: "f32[64]" = _foreach_sqrt_default_1[0]
        getitem_42: "f32[64, 64]" = _foreach_sqrt_default_1[1]
        getitem_43: "f32[64]" = _foreach_sqrt_default_1[2]
        getitem_44: "f32[64, 64]" = _foreach_sqrt_default_1[3]
        getitem_45: "f32[64]" = _foreach_sqrt_default_1[4]
        getitem_46: "f32[64, 64]" = _foreach_sqrt_default_1[5];  _foreach_sqrt_default_1 = None
        return (getitem, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46)


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
