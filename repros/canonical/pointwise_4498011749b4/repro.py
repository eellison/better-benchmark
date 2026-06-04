"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_infer_000
Pattern hash: 4498011749b4
Shape hash: eb18f1d9
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8192, 4096], f32), S([64, 128, 4096]), S([8192, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_190: "f32[8192, 4096]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[64, 128, 4096]" = torch.ops.aten.view.default(addmm_190, _shape_param_0);  addmm_190 = _shape_param_0 = None
        relu_default: "f32[64, 128, 4096]" = torch.ops.aten.relu.default(view_default);  view_default = None
        view_default_1: "f32[8192, 4096]" = torch.ops.aten.view.default(relu_default, _shape_param_1);  relu_default = _shape_param_1 = None
        return view_default_1

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
