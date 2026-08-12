"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: 7155d62f36cd
Shape hash: 12a31f97
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
    def forward(self, arg0_1: "b8[32, 512, 13, 13]", arg1_1: "bf16[32, 512, 13, 13]", arg2_1: "b8[32, 256, 13, 13]", arg3_1: "bf16[]", arg4_1: "b8[32, 256, 13, 13]"):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[32, 512, 13, 13]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mul: "bf16[32, 512, 13, 13]" = torch.ops.aten.mul.Tensor(convert_element_type, 2.0);  convert_element_type = None
        mul_1: "bf16[32, 512, 13, 13]" = torch.ops.aten.mul.Tensor(arg1_1, mul);  arg1_1 = mul = None
        slice_1: "bf16[32, 256, 13, 13]" = torch.ops.aten.slice.Tensor(mul_1, 1, 0, 256)
        slice_2: "bf16[32, 256, 13, 13]" = torch.ops.aten.slice.Tensor(mul_1, 1, 256, 512);  mul_1 = None
        where: "bf16[32, 256, 13, 13]" = torch.ops.aten.where.self(arg2_1, arg3_1, slice_2);  arg2_1 = slice_2 = None
        sum_1: "bf16[256]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        convert_element_type_1: "f32[256]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        where_1: "bf16[32, 256, 13, 13]" = torch.ops.aten.where.self(arg4_1, arg3_1, slice_1);  arg4_1 = arg3_1 = slice_1 = None
        sum_2: "bf16[256]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        convert_element_type_2: "f32[256]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        return (where, convert_element_type_1, where_1, convert_element_type_2)



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
