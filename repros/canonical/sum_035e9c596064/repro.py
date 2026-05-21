"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train_004
Pattern hash: 035e9c596064
Shape hash: b3e05491
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
_shapes_config = "(T([64, 2048, 92], f32), T([64, 2048, 92], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg28_1: "f32[64, 2048, 92]", getitem: "f32[64, 2048, 92]"):
        # No stacktrace found for following nodes
        le_scalar: "b8[64, 2048, 92]" = torch.ops.aten.le.Scalar(arg28_1, 0);  arg28_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[64, 2048, 92]" = torch.ops.aten.where.self(le_scalar, full_default, getitem);  le_scalar = full_default = getitem = None
        sum_dim_int_list: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2]);  where_self = None
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
