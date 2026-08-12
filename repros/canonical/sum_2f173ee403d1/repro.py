"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 2f173ee403d1
Shape hash: 0c938b20
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
    def forward(self, arg0_1: "bf16[512, 128, 128]", arg1_1: "bf16[16, 32, 128, 128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[16, 32, 128, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul: "f32[16, 32, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = None
        sum_1: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        neg: "f32[16, 32, 128, 128]" = torch.ops.aten.neg.default(convert_element_type_1);  convert_element_type_1 = None
        fma: "f32[16, 32, 128, 128]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_2: "bf16[16, 32, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_1: "bf16[512, 128, 128]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
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
