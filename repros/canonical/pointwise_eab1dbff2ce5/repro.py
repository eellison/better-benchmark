"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g77
Pattern hash: eab1dbff2ce5
Shape hash: 3fbf7cc1
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg15_1: "f32[]", arg10_1: "f32[]", arg16_1: "f32[]", arg17_1: "f32[]", arg18_1: "f32[]", arg19_1: "f32[]", arg20_1: "f32[]", arg21_1: "f32[]", arg22_1: "f32[]", arg23_1: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([arg15_1, arg10_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1], 1);  arg15_1 = arg10_1 = arg16_1 = arg17_1 = arg18_1 = arg19_1 = arg20_1 = arg21_1 = arg22_1 = arg23_1 = None
        getitem: "f32[]" = _foreach_add_scalar[0]
        getitem_1: "f32[]" = _foreach_add_scalar[1]
        getitem_2: "f32[]" = _foreach_add_scalar[2]
        getitem_3: "f32[]" = _foreach_add_scalar[3]
        getitem_4: "f32[]" = _foreach_add_scalar[4]
        getitem_5: "f32[]" = _foreach_add_scalar[5]
        getitem_6: "f32[]" = _foreach_add_scalar[6]
        getitem_7: "f32[]" = _foreach_add_scalar[7]
        getitem_8: "f32[]" = _foreach_add_scalar[8]
        getitem_9: "f32[]" = _foreach_add_scalar[9];  _foreach_add_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9)


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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
