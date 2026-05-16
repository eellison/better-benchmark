"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g244
Pattern hash: 5a9cb0c1428e
Shape hash: 3bac544b
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg11_1: "f32[]", arg6_1: "f32[]", arg12_1: "f32[]", arg13_1: "f32[]", arg14_1: "f32[]", arg15_1: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([arg11_1, arg6_1, arg12_1, arg13_1, arg14_1, arg15_1], 1);  arg11_1 = arg6_1 = arg12_1 = arg13_1 = arg14_1 = arg15_1 = None
        getitem: "f32[]" = _foreach_add_scalar[0]
        getitem_1: "f32[]" = _foreach_add_scalar[1]
        getitem_2: "f32[]" = _foreach_add_scalar[2]
        getitem_3: "f32[]" = _foreach_add_scalar[3]
        getitem_4: "f32[]" = _foreach_add_scalar[4]
        getitem_5: "f32[]" = _foreach_add_scalar[5];  _foreach_add_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5)


def _default_make_inputs():
    return [
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
