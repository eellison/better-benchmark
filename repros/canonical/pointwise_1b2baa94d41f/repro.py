"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_GPT2_large_train_002
Pattern hash: 1b2baa94d41f
Shape hash: 6cc11d38
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4, 20, 512, 64], f32), S([-1, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, arg226_1: "f32[4, 20, 512, 64]", _shape_param_0):
        # No stacktrace found for following nodes
        permute_default: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(arg226_1, [0, 2, 1, 3]);  arg226_1 = None
        clone_default: "f32[4, 512, 20, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        _unsafe_view_default: "f32[4, 512, 1280]" = torch.ops.aten._unsafe_view.default(clone_default, [4, 512, 1280]);  clone_default = None
        view_default: "f32[2048, 1280]" = torch.ops.aten.view.default(_unsafe_view_default, _shape_param_0);  _unsafe_view_default = _shape_param_0 = None
        permute_default_1: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
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
