"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_infer_000
Pattern hash: c7ec34b2837b
Shape hash: 04355002
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
_shapes_config = "(T([512, 1280, 1, 1], f32), S([512, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_94: "f32[512, 1280, 1, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        relu_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.relu.default(convolution_94);  convolution_94 = None
        view_default: "f32[512, 1280]" = torch.ops.aten.view.default(relu_default, _shape_param_0);  relu_default = _shape_param_0 = None
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
