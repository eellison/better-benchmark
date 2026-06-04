"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Longformer_infer_003
Pattern hash: b43349159514
Shape hash: 6c893142
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([50265], f16))"

class Repro(torch.nn.Module):
    def forward(self, arg6_1: "f16[50265]"):
        # No stacktrace found for following nodes
        full_default: "f16[7]" = torch.ops.aten.full.default([7], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        cat_default: "f16[50272]" = torch.ops.aten.cat.default([arg6_1, full_default]);  arg6_1 = full_default = None
        return cat_default

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
