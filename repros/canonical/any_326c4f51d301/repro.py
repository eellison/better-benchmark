"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-1-5-linux.aws.a100_graph1
Pattern hash: 326c4f51d301
Shape hash: 445f5bd8
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1, 1024], bf16), S([1024]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        lt_scalar: "b8[1, 1024]" = torch.ops.aten.lt.Scalar(arg0_1, 0)
        gt_scalar: "b8[1, 1024]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
        view_default: "b8[1024]" = torch.ops.aten.view.default(gt_scalar, _shape_param_0);  gt_scalar = _shape_param_0 = None
        any_default: "b8[]" = torch.ops.aten.any.default(view_default);  view_default = None
        return (lt_scalar, any_default)


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
