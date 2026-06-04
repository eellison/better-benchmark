"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train_001
Pattern hash: fa194db93388
Shape hash: f1dac5f3
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 128, 1, 1], f32), T([], f32), T([128, 128, 1, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg187_1: "f32[128, 128, 1, 1]", full_1: "f32[]", getitem_210: "f32[128, 128, 1, 1]"):
        # No stacktrace found for following nodes
        le_scalar: "b8[128, 128, 1, 1]" = torch.ops.aten.le.Scalar(arg187_1, 0);  arg187_1 = None
        where_self: "f32[128, 128, 1, 1]" = torch.ops.aten.where.self(le_scalar, full_1, getitem_210);  le_scalar = full_1 = getitem_210 = None
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3]);  where_self = None
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
