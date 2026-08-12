"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_infer
Pattern hash: 8f076d992692
Shape hash: 0dd51176
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8, 5, 2, 426888]"):
        # No stacktrace found for following nodes
        slice_1: "bf16[8, 4, 2, 426888]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 1, 9223372036854775807);  arg0_1 = None
        slice_2: "bf16[8, 4, 2, 382788]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 382788);  slice_1 = None
        sum_1: "bf16[8, 2, 382788]" = torch.ops.aten.sum.dim_IntList(slice_2, [1])
        return (slice_2, sum_1)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
