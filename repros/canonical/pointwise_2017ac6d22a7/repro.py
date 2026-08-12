"""
Standalone repro captured via capture_hook.
Label: timm_mobilevit_s_train
Pattern hash: 2017ac6d22a7
Shape hash: 259ee10a
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
    def forward(self, arg0_1: "bf16[131072, 144]", arg1_1: "bf16[512, 4, 256, 40]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[512, 256, 144]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[512, 256, 4, 36]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute: "bf16[512, 4, 256, 36]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        slice_scatter: "bf16[512, 4, 256, 40]" = torch.ops.aten.slice_scatter.default(arg1_1, permute, -1, 0, 36);  arg1_1 = permute = None
        return slice_scatter



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
