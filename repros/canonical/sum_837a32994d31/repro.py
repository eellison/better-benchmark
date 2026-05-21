"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train_003
Pattern hash: 837a32994d31
Shape hash: 8b355ac6
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
_shapes_config = "(T([64, 1024, 364], f32), T([64, 1024, 364], b8), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_27: "f32[64, 1024, 364]", arg34_1: "b8[64, 1024, 364]", full_1: "f32[]"):
        # No stacktrace found for following nodes
        full_default: "f32[64, 1024, 372]" = torch.ops.aten.full.default([64, 1024, 372], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default: "f32[64, 1024, 372]" = torch.ops.aten.slice_scatter.default(full_default, getitem_27, 2, 4, -4);  full_default = None
        where_self: "f32[64, 1024, 364]" = torch.ops.aten.where.self(arg34_1, full_1, getitem_27);  arg34_1 = full_1 = getitem_27 = None
        sum_dim_int_list: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2]);  where_self = None
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
