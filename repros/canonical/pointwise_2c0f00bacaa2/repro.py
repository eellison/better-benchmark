"""
Standalone repro captured via capture_hook.
Label: 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_train_010
Pattern hash: 2c0f00bacaa2
Shape hash: 09236877
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([16, 32, 128, 80], f32), S([2048, 2560]))"

class Repro(torch.nn.Module):
    def forward(self, arg21_1: "f32[16, 32, 128, 80]", _shape_param_0):
        # No stacktrace found for following nodes
        permute_default: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(arg21_1, [0, 2, 1, 3]);  arg21_1 = None
        clone_default: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        _unsafe_view_default: "f32[16, 128, 2560]" = torch.ops.aten._unsafe_view.default(clone_default, [16, 128, 2560]);  clone_default = None
        view_default: "f32[2048, 2560]" = torch.ops.aten.view.default(_unsafe_view_default, _shape_param_0);  _unsafe_view_default = _shape_param_0 = None
        return view_default

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
