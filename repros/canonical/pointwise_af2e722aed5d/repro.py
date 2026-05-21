"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train_003
Pattern hash: af2e722aed5d
Shape hash: 40a27608
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
_shapes_config = "(T([1024, 128, 64], f32), S([64, 16, 128, 64]), S([64, 128, -1]), S([8192, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_1: "f32[1024, 128, 64]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[64, 16, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_0);  bmm_1 = _shape_param_0 = None
        permute_default: "f32[64, 128, 16, 64]" = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        clone_default: "f32[64, 128, 16, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1: "f32[64, 128, 1024]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        view_default_2: "f32[8192, 1024]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        return view_default_2



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
