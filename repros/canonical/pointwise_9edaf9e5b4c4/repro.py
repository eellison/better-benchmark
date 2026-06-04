"""
Standalone repro captured via capture_hook.
Label: vllm_facebook_opt-125m_002
Pattern hash: 9edaf9e5b4c4
Shape hash: b9ee2365
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 768], f16), S([4, 512, 768]), S([4, -1, 12, 64]))"

class Repro(torch.nn.Module):
    def forward(self, addmm: "f16[2048, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f16[4, 512, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        mul_tensor: "f16[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_default, 0.125);  view_default = None
        view_default_1: "f16[4, 512, 12, 64]" = torch.ops.aten.view.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None
        permute_default: "f16[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        return permute_default

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
