"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer_001
Pattern hash: 17b234fd7f4f
Shape hash: 0a3b54e6
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
_shapes_config = "(T([8, 1024], f32), S([8192]))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        gt_scalar: "b8[8, 1024]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
        view_default: "b8[8192]" = torch.ops.aten.view.default(gt_scalar, _shape_param_0);  gt_scalar = _shape_param_0 = None
        any_default: "b8[]" = torch.ops.aten.any.default(view_default);  view_default = None
        return any_default



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
