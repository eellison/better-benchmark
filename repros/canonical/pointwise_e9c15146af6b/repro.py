"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: e9c15146af6b
Shape hash: a09a3e2c
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
_shapes_config = "(T([256, 128, 128], f32), T([128], f32), S([32768, 128]))"

class Repro(torch.nn.Module):
    def forward(self, mul_527: "f32[256, 128, 128]", arg5_1: "f32[128]", _shape_param_0):
        # No stacktrace found for following nodes
        mul_tensor: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(mul_527, arg5_1);  mul_527 = arg5_1 = None
        view_default: "f32[32768, 128]" = torch.ops.aten.view.default(mul_tensor, _shape_param_0);  mul_tensor = _shape_param_0 = None
        return view_default



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
