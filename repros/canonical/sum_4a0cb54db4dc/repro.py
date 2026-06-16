"""
Standalone repro captured via capture_hook.
Label: torchbench_lennard_jones_train
Pattern hash: 4a0cb54db4dc
Shape hash: 387810f3
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
    def forward(self, arg0_1: "bf16[128, 16]", arg1_1: "bf16[128, 16]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[128, 16]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type_1, convert_element_type_1);  convert_element_type_1 = None
        sub: "f32[128, 16]" = torch.ops.aten.sub.Tensor(1, mul);  mul = None
        mul_1: "f32[128, 16]" = torch.ops.aten.mul.Tensor(convert_element_type, sub);  convert_element_type = sub = None
        convert_element_type_2: "bf16[128, 16]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16);  mul_1 = None
        permute: "bf16[16, 128]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0])
        sum_1: "f32[1, 16]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0], True, dtype = torch.float32)
        view: "f32[16]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type_3: "bf16[16]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_4: "f32[16]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        return (convert_element_type_2, permute, convert_element_type_4)



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
