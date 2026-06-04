"""
Standalone repro captured via capture_hook.
Label: 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_train_010
Pattern hash: 43aac667469a
Shape hash: eb9a5afd
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 2560], f32), T([2048, 2560], f32), S([16, 128, 2560]), S([16, 128, 2560]))"

class Repro(torch.nn.Module):
    def forward(self, mm_6: "f32[2048, 2560]", mm_8: "f32[2048, 2560]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[16, 128, 2560]" = torch.ops.aten.view.default(mm_6, _shape_param_0);  mm_6 = _shape_param_0 = None
        view_default_1: "f32[16, 128, 2560]" = torch.ops.aten.view.default(mm_8, _shape_param_1);  mm_8 = _shape_param_1 = None
        add_tensor: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        return add_tensor

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
