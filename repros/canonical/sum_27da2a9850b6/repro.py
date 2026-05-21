"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train_003
Pattern hash: 27da2a9850b6
Shape hash: 5b5eea9e
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
_shapes_config = "(T([64, 64, 92844], f32), T([64, 64, 92844], b8))"

class Repro(torch.nn.Module):
    def forward(self, getitem_3: "f32[64, 64, 92844]", arg30_1: "b8[64, 64, 92844]"):
        # No stacktrace found for following nodes
        full_default: "f32[64, 64, 95696]" = torch.ops.aten.full.default([64, 64, 95696], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[64, 64, 95696]" = torch.ops.aten.slice_scatter.default(full_default, getitem_3, 2, 1426, -1426);  full_default = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[64, 64, 92844]" = torch.ops.aten.where.self(arg30_1, full_default_1, getitem_3);  arg30_1 = full_default_1 = getitem_3 = None
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2]);  where_self = None
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
