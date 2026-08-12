"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_infer
Pattern hash: 021e90d9c17d
Shape hash: 6f902f84
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
    def forward(self, arg0_1: "bf16[128, 6, 49, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        permute: "bf16[128, 6, 128, 49]" = torch.ops.aten.permute.default(arg0_1, [0, 1, 3, 2]);  arg0_1 = None
        clone: "bf16[128, 6, 128, 49]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view: "bf16[128, 768, 7, 7]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        return view



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
