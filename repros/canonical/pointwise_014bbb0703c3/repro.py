"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_infer
Pattern hash: 014bbb0703c3
Shape hash: ff8c3966
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
    def forward(self, arg0_1: "bf16[128, 3072, 7, 7]"):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 3072, 7, 7]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        mul: "f32[128, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[128, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.7071067811865476);  convert_element_type = None
        erf: "f32[128, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[128, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[128, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        convert_element_type_1: "bf16[128, 3072, 7, 7]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        return convert_element_type_1



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
