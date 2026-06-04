"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train_003
Pattern hash: f466fd2ff03f
Shape hash: e8467adf
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32, 512, 50257], f32), S([16384, 50257]))"

class Repro(torch.nn.Module):
    def forward(self, arg149_1: "f32[32, 512, 50257]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[16384, 50257]" = torch.ops.aten.view.default(arg149_1, _shape_param_0);  arg149_1 = _shape_param_0 = None
        permute_default: "f32[50257, 16384]" = torch.ops.aten.permute.default(view_default, [1, 0])
        constant_pad_nd_default: "f32[50260, 16384]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 3]);  permute_default = None
        constant_pad_nd_default_1: "f32[16384, 50260]" = torch.ops.aten.constant_pad_nd.default(view_default, [0, 3, 0, 0]);  view_default = None
        return (constant_pad_nd_default, constant_pad_nd_default_1)

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
