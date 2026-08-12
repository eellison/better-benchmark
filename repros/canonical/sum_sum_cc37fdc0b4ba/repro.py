"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: cc37fdc0b4ba
Shape hash: e93c5538
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
    def forward(self, arg0_1: "bf16[512, 960, 7, 7]", arg1_1: "bf16[512, 960, 7, 7]", arg2_1: "bf16[512, 960, 1, 1]"):
        # No stacktrace found for following nodes
        mul: "bf16[512, 960, 7, 7]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[512, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        sum_1: "f32[512, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2, 3], True, dtype = torch.float32);  mul = None
        convert_element_type_1: "bf16[512, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_1, torch.bfloat16);  sum_1 = None
        convert_element_type_2: "f32[512, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        gt: "b8[512, 960, 1, 1]" = torch.ops.aten.gt.Scalar(convert_element_type, -3.0)
        lt: "b8[512, 960, 1, 1]" = torch.ops.aten.lt.Scalar(convert_element_type, 3.0)
        bitwise_and: "b8[512, 960, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt);  gt = lt = None
        mul_1: "f32[512, 960, 1, 1]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.16666666666666666);  convert_element_type_2 = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[512, 960, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_1, full);  bitwise_and = mul_1 = None
        convert_element_type_3: "bf16[512, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        sum_2: "bf16[960]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0, 2, 3])
        convert_element_type_4: "f32[960]" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None
        return (convert_element_type, full, convert_element_type_3, convert_element_type_4)



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
