"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train_008
Pattern hash: 644fbe16a3f2
Shape hash: 214c859f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1, 1024], i64, gen=Index(1026)), T([1, 1024, 1024], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 1024]", arg1_1: "f32[1, 1024, 1024]"):
        # No stacktrace found for following nodes
        eq_scalar: "b8[1, 1024]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default: "b8[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[1, 1024, 1024]" = torch.ops.aten.where.self(unsqueeze_default, full_default, arg1_1);  unsqueeze_default = full_default = arg1_1 = None
        full_default_1: "f32[1026, 1024]" = torch.ops.aten.full.default([1026, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[1026, 1024]" = torch.ops.aten.index_put.default(full_default_1, [arg0_1], where_self, True);  full_default_1 = arg0_1 = where_self = None
        return index_put_default

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
