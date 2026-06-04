"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train_003
Pattern hash: cfa532830ff2
Shape hash: 3ef2db35
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 512, 1452], f32), T([64, 512, 1452], b8), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_21: "f32[64, 512, 1452]", arg33_1: "b8[64, 512, 1452]", full_1: "f32[]"):
        # No stacktrace found for following nodes
        full_default: "f32[64, 512, 1493]" = torch.ops.aten.full.default([64, 512, 1493], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[64, 512, 1493]" = torch.ops.aten.slice_scatter.default(full_default, getitem_21, 2, 20, -21);  full_default = None
        where_self: "f32[64, 512, 1452]" = torch.ops.aten.where.self(arg33_1, full_1, getitem_21);  arg33_1 = full_1 = getitem_21 = None
        sum_dim_int_list: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2]);  where_self = None
        return (slice_scatter_default, sum_dim_int_list)

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
