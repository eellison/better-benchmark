"""
Standalone repro captured via capture_hook.
Label: torchbench_pyhpc_isoneutral_mixing_infer_000
Pattern hash: fd00f02aa65f
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
        full_default: "f64[204, 204, 26]" = torch.ops.aten.full.default([204, 204, 26], 1, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        return full_default

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
