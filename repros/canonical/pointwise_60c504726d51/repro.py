"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train_003
Pattern hash: 60c504726d51
Shape hash: b0ce2fd1
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 6, 1500, 64], f32), S([12000, 384]))"

class Repro(torch.nn.Module):
    def forward(self, arg15_1: "f32[8, 6, 1500, 64]", _shape_param_0):
        # No stacktrace found for following nodes
        permute_default: "f32[8, 1500, 6, 64]" = torch.ops.aten.permute.default(arg15_1, [0, 2, 1, 3]);  arg15_1 = None
        clone_default: "f32[8, 1500, 6, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        _unsafe_view_default: "f32[8, 1500, 384]" = torch.ops.aten._unsafe_view.default(clone_default, [8, 1500, 384]);  clone_default = None
        view_default: "f32[12000, 384]" = torch.ops.aten.view.default(_unsafe_view_default, _shape_param_0);  _unsafe_view_default = _shape_param_0 = None
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
