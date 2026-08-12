"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: 1edbc54f25f6
Shape hash: 44864090
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
    def forward(self, arg0_1: "b8[128, 9216]", arg1_1: "bf16[128, 9216]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[128, 9216]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mul: "bf16[128, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
        mul_1: "bf16[128, 9216]" = torch.ops.aten.mul.Tensor(arg1_1, mul);  arg1_1 = mul = None
        view: "bf16[128, 256, 6, 6]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
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
