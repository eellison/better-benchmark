"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g20
Pattern hash: 44ba1c4090a0
Shape hash: e5e7892d
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
    def forward(self, addmm_44: "f16[1584, 2304]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        reshape_default: "f16[8, 198, 2304]" = torch.ops.aten.reshape.default(addmm_44, _shape_param_0);  addmm_44 = _shape_param_0 = None
        reshape_default_1: "f16[8, 198, 3, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f16[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None
        unbind_int = torch.ops.aten.unbind.int(permute_default);  permute_default = None
        getitem: "f16[8, 12, 198, 64]" = unbind_int[0]
        getitem_1: "f16[8, 12, 198, 64]" = unbind_int[1]
        getitem_2: "f16[8, 12, 198, 64]" = unbind_int[2];  unbind_int = None
        return (getitem, getitem_1, getitem_2)


def _default_make_inputs():
    return [
    torch.randn([1584, 2304], dtype=torch.float16, device='cuda'),
    [8, 198, 2304],  # _shape_param_0
    [8, 198, 3, 12, 64],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
