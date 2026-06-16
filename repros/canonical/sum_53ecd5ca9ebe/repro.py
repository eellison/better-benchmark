"""
Standalone repro captured via capture_hook.
Label: torchbench_nvidia_deeprecommender_train
Pattern hash: 53ecd5ca9ebe
Shape hash: a7f39cdb
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
    def forward(self, arg0_1: "bf16[256, 512]", arg1_1: "bf16[256, 512]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[256, 512]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        le: "b8[256, 512]" = torch.ops.aten.le.Scalar(convert_element_type_1, 0)
        mul: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, 1)
        mul_1: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul, 1.7580993408473766);  mul = None
        mul_2: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        exp: "f32[256, 512]" = torch.ops.aten.exp.default(mul_2);  mul_2 = None
        mul_3: "f32[256, 512]" = torch.ops.aten.mul.Tensor(mul_1, exp);  mul_1 = exp = None
        mul_4: "f32[256, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.0507009873554805);  convert_element_type = None
        where: "f32[256, 512]" = torch.ops.aten.where.self(le, mul_3, mul_4);  le = mul_3 = mul_4 = None
        convert_element_type_2: "bf16[256, 512]" = torch.ops.prims.convert_element_type.default(where, torch.bfloat16);  where = None
        permute: "bf16[512, 256]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0])
        sum_1: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0], True, dtype = torch.float32)
        view: "f32[512]" = torch.ops.aten.view.default(sum_1, _shape_param_0);  sum_1 = _shape_param_0 = None
        convert_element_type_3: "bf16[512]" = torch.ops.prims.convert_element_type.default(view, torch.bfloat16);  view = None
        convert_element_type_4: "f32[512]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
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
