"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g77
Pattern hash: e5a5dadc36ce
Shape hash: 87f4a5ad
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_30: "f32[16, 1]", getitem_31: "f32[16]", getitem_32: "f32[16, 16]", getitem_33: "f32[16]", getitem_34: "f32[16, 16]", getitem_35: "f32[16]", getitem_36: "f32[16, 16]", getitem_37: "f32[16]", getitem_38: "f32[1, 16]", getitem_39: "f32[1]", arg40_1: "f32[16, 1]", arg41_1: "f32[16]", arg42_1: "f32[16, 16]", arg43_1: "f32[16]", arg44_1: "f32[16, 16]", arg45_1: "f32[16]", arg46_1: "f32[16, 16]", arg47_1: "f32[16]", arg48_1: "f32[1, 16]", arg49_1: "f32[1]", getitem: "f32[]", getitem_1: "f32[]", getitem_2: "f32[]", getitem_3: "f32[]", getitem_4: "f32[]", getitem_5: "f32[]", getitem_6: "f32[]", getitem_7: "f32[]", getitem_8: "f32[]", getitem_9: "f32[]", getitem_80: "f32[]", getitem_81: "f32[]", getitem_82: "f32[]", getitem_83: "f32[]", getitem_84: "f32[]", getitem_85: "f32[]", getitem_86: "f32[]", getitem_87: "f32[]", getitem_88: "f32[]", getitem_89: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39], [arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1], [arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1], 0.0010000000000000009);  getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = arg40_1 = arg41_1 = arg42_1 = arg43_1 = arg44_1 = arg45_1 = arg46_1 = arg47_1 = arg48_1 = arg49_1 = None
        getitem: "f32[16, 1]" = _foreach_addcmul_scalar[0]
        getitem_40: "f32[16]" = _foreach_addcmul_scalar[1]
        getitem_41: "f32[16, 16]" = _foreach_addcmul_scalar[2]
        getitem_42: "f32[16]" = _foreach_addcmul_scalar[3]
        getitem_43: "f32[16, 16]" = _foreach_addcmul_scalar[4]
        getitem_44: "f32[16]" = _foreach_addcmul_scalar[5]
        getitem_45: "f32[16, 16]" = _foreach_addcmul_scalar[6]
        getitem_46: "f32[16]" = _foreach_addcmul_scalar[7]
        getitem_47: "f32[1, 16]" = _foreach_addcmul_scalar[8]
        getitem_48: "f32[1]" = _foreach_addcmul_scalar[9];  _foreach_addcmul_scalar = None
        getitem_49 = getitem
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem_49, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9]);  getitem_49 = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = None
        getitem_10: "f32[]" = _foreach_pow_scalar_and_tensor[0]
        getitem_11: "f32[]" = _foreach_pow_scalar_and_tensor[1]
        getitem_12: "f32[]" = _foreach_pow_scalar_and_tensor[2]
        getitem_13: "f32[]" = _foreach_pow_scalar_and_tensor[3]
        getitem_14: "f32[]" = _foreach_pow_scalar_and_tensor[4]
        getitem_15: "f32[]" = _foreach_pow_scalar_and_tensor[5]
        getitem_16: "f32[]" = _foreach_pow_scalar_and_tensor[6]
        getitem_17: "f32[]" = _foreach_pow_scalar_and_tensor[7]
        getitem_18: "f32[]" = _foreach_pow_scalar_and_tensor[8]
        getitem_19: "f32[]" = _foreach_pow_scalar_and_tensor[9];  _foreach_pow_scalar_and_tensor = None
        _foreach_neg_default = torch.ops.aten._foreach_neg.default([getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89]);  getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = None
        getitem_90: "f32[]" = _foreach_neg_default[0]
        getitem_91: "f32[]" = _foreach_neg_default[1]
        getitem_92: "f32[]" = _foreach_neg_default[2]
        getitem_93: "f32[]" = _foreach_neg_default[3]
        getitem_94: "f32[]" = _foreach_neg_default[4]
        getitem_95: "f32[]" = _foreach_neg_default[5]
        getitem_96: "f32[]" = _foreach_neg_default[6]
        getitem_97: "f32[]" = _foreach_neg_default[7]
        getitem_98: "f32[]" = _foreach_neg_default[8]
        getitem_99: "f32[]" = _foreach_neg_default[9];  _foreach_neg_default = None
        return (getitem, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99)


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
    torch.tensor(1),  # getitem_49 (unknown shape)
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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
