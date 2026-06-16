"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_infer
Pattern hash: 1918882eece2
Shape hash: a7e138d5
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[8, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        gt: "b8[8, 1024]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
        view: "b8[8192]" = torch.ops.aten.view.default(gt, _shape_param_0);  _shape_param_0 = None
        any_1: "b8[]" = torch.ops.aten.any.default(view);  view = None
        return (gt, any_1)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
