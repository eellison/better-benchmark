"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer_000
Pattern hash: 9efa43391431
Shape hash: d7517139
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "()"

class Repro(torch.nn.Module):
    def forward(self):
        # No stacktrace found for following nodes
        full_default: "f16[1, 1, 1, 4096]" = torch.ops.aten.full.default([1, 1, 1, 4096], -0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_int: "f16[1, 1, 4096]" = torch.ops.aten.select.int(full_default, 1, 0);  full_default = None
        select_int_1: "f16[1, 4096]" = torch.ops.aten.select.int(select_int, 1, 0);  select_int = None
        return select_int_1

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
