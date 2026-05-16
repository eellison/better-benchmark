"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g244
Pattern hash: bd61ace5fc2f
Shape hash: af0f5991
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg10_1: "f32[64]", arg8_1: "f32[64, 64]", arg20_1: "f32[64]", arg21_1: "f32[64, 64]", arg22_1: "f32[64]", arg23_1: "f32[64, 64]", getitem_36: "f32[]", getitem_37: "f32[]", getitem_38: "f32[]", getitem_39: "f32[]", getitem_40: "f32[]", getitem_41: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_mul_scalar = torch.ops.aten._foreach_mul.Scalar([arg10_1, arg8_1, arg20_1, arg21_1, arg22_1, arg23_1], 0.999);  arg10_1 = arg8_1 = arg20_1 = arg21_1 = arg22_1 = arg23_1 = None
        getitem: "f32[64]" = _foreach_mul_scalar[0]
        getitem_1: "f32[64, 64]" = _foreach_mul_scalar[1]
        getitem_2: "f32[64]" = _foreach_mul_scalar[2]
        getitem_3: "f32[64, 64]" = _foreach_mul_scalar[3]
        getitem_4: "f32[64]" = _foreach_mul_scalar[4]
        getitem_5: "f32[64, 64]" = _foreach_mul_scalar[5];  _foreach_mul_scalar = None
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41], 1);  getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = None
        getitem_42: "f32[]" = _foreach_sub_scalar[0]
        getitem_43: "f32[]" = _foreach_sub_scalar[1]
        getitem_44: "f32[]" = _foreach_sub_scalar[2]
        getitem_45: "f32[]" = _foreach_sub_scalar[3]
        getitem_46: "f32[]" = _foreach_sub_scalar[4]
        getitem_47: "f32[]" = _foreach_sub_scalar[5];  _foreach_sub_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47)


def _default_make_inputs():
    return [
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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
