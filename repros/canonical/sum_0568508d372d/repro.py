"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train_001
Pattern hash: 0568508d372d
Shape hash: d0a75f23
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
_shapes_config = "(T([1024, 4096], f32), T([1024, 4096], f32), S([4096]))"

class Repro(torch.nn.Module):
    def forward(self, arg21_1: "f32[1024, 4096]", mm: "f32[1024, 4096]", _shape_param_0):
        # No stacktrace found for following nodes
        le_scalar: "b8[1024, 4096]" = torch.ops.aten.le.Scalar(arg21_1, 0);  arg21_1 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1024, 4096]" = torch.ops.aten.where.self(le_scalar, full_default, mm);  le_scalar = full_default = mm = None
        permute_default: "f32[4096, 1024]" = torch.ops.aten.permute.default(where_self, [1, 0])
        sum_dim_int_list: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where_self, [0], True);  where_self = None
        view_default: "f32[4096]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        return (permute_default, view_default)



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
