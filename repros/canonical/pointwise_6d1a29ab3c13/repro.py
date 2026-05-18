"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s1_g43
Pattern hash: 6d1a29ab3c13
Shape hash: 15059ffd
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
    def forward(self, arg0_1: "i64[2, 200000]"):
        # No stacktrace found for following nodes
        select_int: "i64[200000]" = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_int_1: "i64[200000]" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        ne_tensor: "b8[200000]" = torch.ops.aten.ne.Tensor(select_int, select_int_1);  select_int = select_int_1 = None
        iota_default: "i64[10000]" = torch.ops.prims.iota.default(10000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default: "i64[1, 10000]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None
        repeat_default: "i64[2, 10000]" = torch.ops.aten.repeat.default(unsqueeze_default, [2, 1]);  unsqueeze_default = None
        return (ne_tensor, repeat_default)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [2, 200000], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
