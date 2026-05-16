"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g77
Pattern hash: 0cc71f31119e
Shape hash: 7286d40a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg14_1: "f32[16, 1]", arg12_1: "f32[16]", arg32_1: "f32[16, 16]", arg33_1: "f32[16]", arg34_1: "f32[16, 16]", arg35_1: "f32[16]", arg36_1: "f32[16, 16]", arg37_1: "f32[16]", arg38_1: "f32[1, 16]", arg39_1: "f32[1]", getitem_60: "f32[]", getitem_61: "f32[]", getitem_62: "f32[]", getitem_63: "f32[]", getitem_64: "f32[]", getitem_65: "f32[]", getitem_66: "f32[]", getitem_67: "f32[]", getitem_68: "f32[]", getitem_69: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_mul_scalar = torch.ops.aten._foreach_mul.Scalar([arg14_1, arg12_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1], 0.999);  arg14_1 = arg12_1 = arg32_1 = arg33_1 = arg34_1 = arg35_1 = arg36_1 = arg37_1 = arg38_1 = arg39_1 = None
        getitem: "f32[16, 1]" = _foreach_mul_scalar[0]
        getitem_1: "f32[16]" = _foreach_mul_scalar[1]
        getitem_2: "f32[16, 16]" = _foreach_mul_scalar[2]
        getitem_3: "f32[16]" = _foreach_mul_scalar[3]
        getitem_4: "f32[16, 16]" = _foreach_mul_scalar[4]
        getitem_5: "f32[16]" = _foreach_mul_scalar[5]
        getitem_6: "f32[16, 16]" = _foreach_mul_scalar[6]
        getitem_7: "f32[16]" = _foreach_mul_scalar[7]
        getitem_8: "f32[1, 16]" = _foreach_mul_scalar[8]
        getitem_9: "f32[1]" = _foreach_mul_scalar[9];  _foreach_mul_scalar = None
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69], 1);  getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = None
        getitem_70: "f32[]" = _foreach_sub_scalar[0]
        getitem_71: "f32[]" = _foreach_sub_scalar[1]
        getitem_72: "f32[]" = _foreach_sub_scalar[2]
        getitem_73: "f32[]" = _foreach_sub_scalar[3]
        getitem_74: "f32[]" = _foreach_sub_scalar[4]
        getitem_75: "f32[]" = _foreach_sub_scalar[5]
        getitem_76: "f32[]" = _foreach_sub_scalar[6]
        getitem_77: "f32[]" = _foreach_sub_scalar[7]
        getitem_78: "f32[]" = _foreach_sub_scalar[8]
        getitem_79: "f32[]" = _foreach_sub_scalar[9];  _foreach_sub_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79)


def _default_make_inputs():
    return [
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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
