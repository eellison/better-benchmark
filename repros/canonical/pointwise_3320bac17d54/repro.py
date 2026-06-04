"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer_009
Pattern hash: 3320bac17d54
Shape hash: e92a1ac3
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1], i64))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "i64[1]"):
        # No stacktrace found for following nodes
        select_int: "i64[]" = torch.ops.aten.select.int(arg2_1, 0, 0)
        full_default: "i64[]" = torch.ops.aten.full.default([], 32, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default: "i64[]" = torch.ops.aten.copy.default(select_int, full_default);  select_int = full_default = None
        select_scatter_default: "i64[1]" = torch.ops.aten.select_scatter.default(arg2_1, copy_default, 0, 0);  copy_default = None
        copy__default: "i64[1]" = torch.ops.aten.copy_.default(arg2_1, select_scatter_default);  arg2_1 = select_scatter_default = None
        return copy__default

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
