"""
Standalone repro captured via capture_hook.
Label: torchbench_nvidia_deeprecommender_train
Pattern hash: 7039286a336e
Shape hash: 35648b40
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
    def forward(self, arg0_1: "b8[256, 1024]", arg1_1: "bf16[256, 1024]", arg2_1: "bf16[256, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        mul: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type, 5.000000000000001);  convert_element_type = None
        mul_1: "bf16[256, 1024]" = torch.ops.aten.mul.Tensor(arg1_1, mul);  arg1_1 = mul = None
        convert_element_type_1: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(mul_1, torch.float32);  mul_1 = None
        convert_element_type_2: "f32[256, 1024]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        le: "b8[256, 1024]" = torch.ops.aten.le.Scalar(convert_element_type_2, 0)
        mul_2: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1)
        mul_3: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(mul_2, 1.7580993408473766);  mul_2 = None
        mul_4: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        exp: "f32[256, 1024]" = torch.ops.aten.exp.default(mul_4);  mul_4 = None
        mul_5: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(mul_3, exp);  mul_3 = exp = None
        mul_6: "f32[256, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.0507009873554805);  convert_element_type_1 = None
        where: "f32[256, 1024]" = torch.ops.aten.where.self(le, mul_5, mul_6);  le = mul_5 = mul_6 = None
        convert_element_type_3: "bf16[256, 1024]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        permute: "bf16[1024, 256]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0])
        sum_1: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0], True, dtype = torch.float32)
        view: "f32[1024]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type_4: "bf16[1024]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_5: "f32[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        return (convert_element_type_3, permute, convert_element_type_5)



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
