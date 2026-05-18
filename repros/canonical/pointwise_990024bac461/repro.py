"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s5_g77
Pattern hash: 990024bac461
Shape hash: 3fbf7cc1
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
    def forward(self, getitem: "f32[]", getitem_1: "f32[]", getitem_2: "f32[]", getitem_3: "f32[]", getitem_4: "f32[]", getitem_5: "f32[]", getitem_6: "f32[]", getitem_7: "f32[]", getitem_8: "f32[]", getitem_9: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.999, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9]);  getitem = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = None
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
        return (getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19)


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
