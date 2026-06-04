"""
Standalone repro captured via capture_hook.
Label: torchbench_lennard_jones_train_001
Pattern hash: d7cc0afd5743
Shape hash: 272070d5
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1000, 1], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg9_1: "f32[1000, 1]"):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(arg9_1, [0], True);  arg9_1 = None
        view_default: "f32[1]" = torch.ops.aten.view.default(sum_dim_int_list, [1]);  sum_dim_int_list = None
        return view_default

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
