"""
Standalone repro captured via capture_hook.
Label: timm_deit_tiny_patch16_224.fb_in1k_train_001
Pattern hash: baa1198c62c0
Shape hash: 8445780e
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 3, 197, 64], f32), S([25216, 192]))"

class Repro(torch.nn.Module):
    def forward(self, arg84_1: "f32[128, 3, 197, 64]", _shape_param_0):
        # No stacktrace found for following nodes
        permute_default: "f32[128, 197, 3, 64]" = torch.ops.aten.permute.default(arg84_1, [0, 2, 1, 3]);  arg84_1 = None
        clone_default: "f32[128, 197, 3, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        _unsafe_view_default: "f32[128, 197, 192]" = torch.ops.aten._unsafe_view.default(clone_default, [128, 197, 192]);  clone_default = None
        view_default: "f32[25216, 192]" = torch.ops.aten.view.default(_unsafe_view_default, _shape_param_0);  _unsafe_view_default = _shape_param_0 = None
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
