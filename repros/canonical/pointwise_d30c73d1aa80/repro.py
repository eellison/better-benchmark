"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g244
Pattern hash: d30c73d1aa80
Shape hash: fb8c00e9
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
    def forward(self, arg0_1: "f32[64]", arg1_1: "f32[64, 64]", arg2_1: "f32[64]", arg3_1: "f32[64, 64]", arg4_1: "f32[64]", arg5_1: "f32[64, 64]", getitem_12: "f32[64]", getitem_13: "f32[64, 64]", getitem_14: "f32[64]", getitem_15: "f32[64, 64]", getitem_16: "f32[64]", getitem_17: "f32[64, 64]", getitem_96: "f32[64]", getitem_97: "f32[64, 64]", getitem_98: "f32[64]", getitem_99: "f32[64, 64]", getitem_100: "f32[64]", getitem_101: "f32[64, 64]"):
        # No stacktrace found for following nodes
        _foreach_addcdiv_scalar = torch.ops.aten._foreach_addcdiv.Scalar([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1], [getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17], [getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101]);  arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg5_1 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_96 = getitem_97 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = None
        getitem: "f32[64]" = _foreach_addcdiv_scalar[0]
        getitem_102: "f32[64, 64]" = _foreach_addcdiv_scalar[1]
        getitem_103: "f32[64]" = _foreach_addcdiv_scalar[2]
        getitem_104: "f32[64, 64]" = _foreach_addcdiv_scalar[3]
        getitem_105: "f32[64]" = _foreach_addcdiv_scalar[4]
        getitem_106: "f32[64, 64]" = _foreach_addcdiv_scalar[5];  _foreach_addcdiv_scalar = None
        return (getitem, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106)


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
