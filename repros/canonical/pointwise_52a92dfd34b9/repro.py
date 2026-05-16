"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g77
Pattern hash: 52a92dfd34b9
Shape hash: 47e610c5
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[16, 1]", arg1_1: "f32[16]", arg2_1: "f32[16, 16]", arg3_1: "f32[16]", arg4_1: "f32[16, 16]", arg5_1: "f32[16]", arg6_1: "f32[16, 16]", arg7_1: "f32[16]", arg8_1: "f32[1, 16]", arg9_1: "f32[1]", getitem_20: "f32[16, 1]", getitem_21: "f32[16]", getitem_22: "f32[16, 16]", getitem_23: "f32[16]", getitem_24: "f32[16, 16]", getitem_25: "f32[16]", getitem_26: "f32[16, 16]", getitem_27: "f32[16]", getitem_28: "f32[1, 16]", getitem_29: "f32[1]", getitem_160: "f32[16, 1]", getitem_161: "f32[16]", getitem_162: "f32[16, 16]", getitem_163: "f32[16]", getitem_164: "f32[16, 16]", getitem_165: "f32[16]", getitem_166: "f32[16, 16]", getitem_167: "f32[16]", getitem_168: "f32[1, 16]", getitem_169: "f32[1]"):
        # No stacktrace found for following nodes
        _foreach_addcdiv_scalar = torch.ops.aten._foreach_addcdiv.Scalar([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1], [getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29], [getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169]);  arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg5_1 = arg6_1 = arg7_1 = arg8_1 = arg9_1 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_160 = getitem_161 = getitem_162 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_167 = getitem_168 = getitem_169 = None
        getitem: "f32[16, 1]" = _foreach_addcdiv_scalar[0]
        getitem_170: "f32[16]" = _foreach_addcdiv_scalar[1]
        getitem_171: "f32[16, 16]" = _foreach_addcdiv_scalar[2]
        getitem_172: "f32[16]" = _foreach_addcdiv_scalar[3]
        getitem_173: "f32[16, 16]" = _foreach_addcdiv_scalar[4]
        getitem_174: "f32[16]" = _foreach_addcdiv_scalar[5]
        getitem_175: "f32[16, 16]" = _foreach_addcdiv_scalar[6]
        getitem_176: "f32[16]" = _foreach_addcdiv_scalar[7]
        getitem_177: "f32[1, 16]" = _foreach_addcdiv_scalar[8]
        getitem_178: "f32[1]" = _foreach_addcdiv_scalar[9];  _foreach_addcdiv_scalar = None
        return (getitem, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178)


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
