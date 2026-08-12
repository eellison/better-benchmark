"""
Standalone repro captured via capture_hook.
Label: timm_vit_base_patch16_siglip_256_infer
Pattern hash: 93756b266db9
Shape hash: a6d7e457
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
    def forward(self, arg0_1: "bf16[128, 768]", arg1_1: "bf16[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[128, 1, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "bf16[128, 1, 768]" = torch.ops.aten.add.Tensor(view, arg1_1);  view = arg1_1 = None
        view_1: "bf16[128, 1, 12, 64]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        permute: "bf16[128, 12, 1, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        return permute



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
