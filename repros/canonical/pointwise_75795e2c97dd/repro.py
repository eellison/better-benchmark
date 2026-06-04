"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer_000
Pattern hash: 75795e2c97dd
Shape hash: 0931dbc0
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 1024], i64))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[8, 1024]"):
        # No stacktrace found for following nodes
        ne_scalar: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg0_1, 1);  arg0_1 = None
        convert_element_type_default: "i32[8, 1024]" = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        return convert_element_type_default

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
