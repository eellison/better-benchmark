"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer
Pattern hash: 182f6f9450b9
Shape hash: 5e1cd0cb
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
    def forward(self, arg0_1: "bf16[128, 1152, 1, 1]", arg1_1: "bf16[128, 1152, 7, 7]"):
        # No stacktrace found for following nodes
        sigmoid: "bf16[128, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(arg0_1);  arg0_1 = None
        mul: "bf16[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(arg1_1, sigmoid);  arg1_1 = sigmoid = None
        return mul



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
