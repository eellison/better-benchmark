"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g244
Pattern hash: fe0ddf8d5cdc
Shape hash: c9b2e66a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_18: "f32[64]", getitem_19: "f32[64, 64]", getitem_20: "f32[64]", getitem_21: "f32[64, 64]", getitem_22: "f32[64]", getitem_23: "f32[64, 64]", arg24_1: "f32[64]", arg25_1: "f32[64, 64]", arg26_1: "f32[64]", arg27_1: "f32[64, 64]", arg28_1: "f32[64]", arg29_1: "f32[64, 64]", getitem: "f32[]", getitem_1: "f32[]", getitem_2: "f32[]", getitem_3: "f32[]", getitem_4: "f32[]", getitem_5: "f32[]", getitem_48: "f32[]", getitem_49: "f32[]", getitem_50: "f32[]", getitem_51: "f32[]", getitem_52: "f32[]", getitem_53: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23], [arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1], [arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1], 0.0010000000000000009);  getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = None
        getitem: "f32[64]" = _foreach_addcmul_scalar[0]
        getitem_24: "f32[64, 64]" = _foreach_addcmul_scalar[1]
        getitem_25: "f32[64]" = _foreach_addcmul_scalar[2]
        getitem_26: "f32[64, 64]" = _foreach_addcmul_scalar[3]
        getitem_27: "f32[64]" = _foreach_addcmul_scalar[4]
        getitem_28: "f32[64, 64]" = _foreach_addcmul_scalar[5];  _foreach_addcmul_scalar = None
        getitem_29 = getitem
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem_29, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5]);  getitem_29 = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = None
        getitem_6: "f32[]" = _foreach_pow_scalar_and_tensor[0]
        getitem_7: "f32[]" = _foreach_pow_scalar_and_tensor[1]
        getitem_8: "f32[]" = _foreach_pow_scalar_and_tensor[2]
        getitem_9: "f32[]" = _foreach_pow_scalar_and_tensor[3]
        getitem_10: "f32[]" = _foreach_pow_scalar_and_tensor[4]
        getitem_11: "f32[]" = _foreach_pow_scalar_and_tensor[5];  _foreach_pow_scalar_and_tensor = None
        _foreach_neg_default = torch.ops.aten._foreach_neg.default([getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53]);  getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = None
        getitem_54: "f32[]" = _foreach_neg_default[0]
        getitem_55: "f32[]" = _foreach_neg_default[1]
        getitem_56: "f32[]" = _foreach_neg_default[2]
        getitem_57: "f32[]" = _foreach_neg_default[3]
        getitem_58: "f32[]" = _foreach_neg_default[4]
        getitem_59: "f32[]" = _foreach_neg_default[5];  _foreach_neg_default = None
        return (getitem, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59)


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
    torch.tensor(1),  # getitem_29 (unknown shape)
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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
