"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: d14b9f62e2e6
Shape hash: 0fc3d1e9
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
    def forward(self, arg0_1: "bf16[768, 196, 64]", arg1_1: "bf16[768, 64, 196]", arg2_1: "bf16[768, 196, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        view: "bf16[128, 6, 196, 64]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        view_1: "bf16[128, 6, 64, 196]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        view_2: "bf16[128, 6, 196, 64]" = torch.ops.aten.view.default(arg2_1, _shape_param_2);  arg2_1 = _shape_param_2 = None
        permute: "bf16[128, 6, 196, 64]" = torch.ops.aten.permute.default(view_1, [0, 1, 3, 2]);  view_1 = None
        cat: "bf16[384, 6, 196, 64]" = torch.ops.aten.cat.default([view_2, permute, view]);  view_2 = permute = view = None
        view_3: "bf16[3, 128, 6, 196, 64]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        permute_1: "bf16[128, 3, 6, 64, 196]" = torch.ops.aten.permute.default(view_3, [1, 0, 2, 4, 3]);  view_3 = None
        clone: "bf16[128, 3, 6, 64, 196]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_4: "bf16[128, 1152, 14, 14]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        return view_4



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
