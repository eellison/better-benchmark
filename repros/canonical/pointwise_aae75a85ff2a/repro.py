"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train_004
Pattern hash: aae75a85ff2a
Shape hash: 3d5038f5
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 1, 1024, 1024], b8), S([8, 16, 1024, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg7_1: "b8[8, 1, 1024, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(arg7_1, full_default_1, full_default);  arg7_1 = full_default_1 = full_default = None
        expand_default: "f32[8, 16, 1024, 1024]" = torch.ops.aten.expand.default(where_self, _shape_param_0);  where_self = _shape_param_0 = None
        return expand_default

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
