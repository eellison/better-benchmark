"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_infer_000
Pattern hash: 5ccd045f4193
Shape hash: 243e759c
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([12000, 256], f16), S([8, 1500, 256]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_20: "f16[12000, 256]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f16[8, 1500, 256]" = torch.ops.aten.view.default(addmm_20, _shape_param_0);  addmm_20 = _shape_param_0 = None
        mean_dim: "f16[8, 256]" = torch.ops.aten.mean.dim(view_default, [1]);  view_default = None
        return mean_dim

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
