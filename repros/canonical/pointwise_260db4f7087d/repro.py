"""
Standalone repro captured via capture_hook.
Label: hf_XLNetLMHeadModel_train
Pattern hash: 260db4f7087d
Shape hash: 0578bbc7
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
    def forward(self, arg0_1: "bf16[16, 16, 512, 1023]", arg1_1: "bf16[16, 16, 1024, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[16, 16, 1023, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        slice_scatter: "bf16[16, 16, 1024, 512]" = torch.ops.aten.slice_scatter.default(arg1_1, view, 2, 1, 9223372036854775807);  arg1_1 = view = None
        view_1: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(slice_scatter, _shape_param_1);  slice_scatter = _shape_param_1 = None
        view_2: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.permute.default(view_2, [0, 1, 2, 4, 3]);  view_2 = None
        view_3: "bf16[256, 512, 1024]" = torch.ops.aten.view.default(permute, _shape_param_3);  permute = _shape_param_3 = None
        return view_3



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
