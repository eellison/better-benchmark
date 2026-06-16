"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: c827b10bd5dd
Shape hash: d100005c
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
    def forward(self, arg0_1: "bf16[768, 200, 64]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        slice_1: "bf16[768, 196, 64]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, -4);  arg0_1 = None
        view: "bf16[128, 6, 196, 64]" = torch.ops.aten.view.default(slice_1, _shape_param_0);  slice_1 = _shape_param_0 = None
        permute: "bf16[128, 6, 64, 196]" = torch.ops.aten.permute.default(view, [0, 1, 3, 2]);  view = None
        clone: "bf16[128, 6, 64, 196]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[128, 384, 14, 14]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        return view_1



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
