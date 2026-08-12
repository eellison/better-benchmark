"""
Standalone repro captured via capture_hook.
Label: torchbench_dcgan_train
Pattern hash: e8ae58ea3c65
Shape hash: 8ebdd403
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
    def forward(self, arg0_1: "bf16[32, 1, 1, 1]", arg1_1: "bf16[32, 1, 1, 1]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[32, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[32, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        sub: "f32[32, 1, 1, 1]" = torch.ops.aten.sub.Tensor(1, convert_element_type_1)
        mul: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_1, sub);  convert_element_type_1 = sub = None
        mul_1: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        convert_element_type_2: "bf16[32, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        return convert_element_type_2



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
