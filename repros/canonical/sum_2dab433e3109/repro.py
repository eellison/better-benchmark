"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train_004
Pattern hash: 2dab433e3109
Shape hash: e5553518
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 64, 95696], f32), T([], f32), T([64, 64, 95696], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg13_1: "f32[64, 64, 95696]", full: "f32[]", getitem_30: "f32[64, 64, 95696]"):
        # No stacktrace found for following nodes
        le_scalar: "b8[64, 64, 95696]" = torch.ops.aten.le.Scalar(arg13_1, 0);  arg13_1 = None
        where_self: "f32[64, 64, 95696]" = torch.ops.aten.where.self(le_scalar, full, getitem_30);  le_scalar = full = getitem_30 = None
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2]);  where_self = None
        return sum_dim_int_list



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
