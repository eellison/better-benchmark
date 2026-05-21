"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer_000
Pattern hash: d6f684372ab8
Shape hash: 8658424f
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
_shapes_config = "(T([32, 32, 8, 32, 2], f32), S([32, 32, 8, 64]), S([32, 8, 32, 64]), S([256, 32, 64]))"

class Repro(torch.nn.Module):
    def forward(self, view_as_real_14: "f32[32, 32, 8, 32, 2]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[32, 32, 8, 64]" = torch.ops.aten.view.default(view_as_real_14, _shape_param_0);  view_as_real_14 = _shape_param_0 = None
        permute_default: "f32[32, 8, 32, 64]" = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        expand_default: "f32[32, 8, 32, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        clone_default: "f32[32, 8, 32, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_1: "f32[256, 32, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        return view_default_1



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
