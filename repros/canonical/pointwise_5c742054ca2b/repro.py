"""
Standalone repro captured via capture_hook.
Label: hf_GPT2ForSequenceClassification_train_001
Pattern hash: 5c742054ca2b
Shape hash: 10caa1e1
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 12, 1024, 64], f32), S([-1, 768]))"

class Repro(torch.nn.Module):
    def forward(self, arg86_1: "f32[8, 12, 1024, 64]", _shape_param_0):
        # No stacktrace found for following nodes
        permute_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(arg86_1, [0, 2, 1, 3]);  arg86_1 = None
        clone_default: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        _unsafe_view_default: "f32[8, 1024, 768]" = torch.ops.aten._unsafe_view.default(clone_default, [8, 1024, 768]);  clone_default = None
        view_default: "f32[8192, 768]" = torch.ops.aten.view.default(_unsafe_view_default, _shape_param_0);  _unsafe_view_default = _shape_param_0 = None
        permute_default_1: "f32[768, 8192]" = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        return permute_default_1

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
