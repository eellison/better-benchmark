"""
Standalone repro captured via capture_hook.
Label: genai_SoftmaxBackward_static
Pattern hash: 4fd6e4019857
Shape hash: 74d25999
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
    def forward(self, arg0_1: "bf16[8192, 262144]", arg1_1: "f32[8192, 1]", arg2_1: "f32[8192, 1]", arg3_1: "bf16[]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        sub: "f32[8192, 262144]" = torch.ops.aten.sub.Tensor(convert_element_type, arg1_1);  convert_element_type = arg1_1 = None
        exp: "f32[8192, 262144]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[8192, 262144]" = torch.ops.aten.div.Tensor(exp, arg2_1);  exp = arg2_1 = None
        convert_element_type_1: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        convert_element_type_2: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        neg: "f32[8192, 262144]" = torch.ops.aten.neg.default(convert_element_type_2)
        expand: "bf16[8192, 262144]" = torch.ops.aten.expand.default(arg3_1, _shape_param_0);  arg3_1 = _shape_param_0 = None
        convert_element_type_3: "f32[8192, 262144]" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None
        mul: "f32[8192, 262144]" = torch.ops.aten.mul.Tensor(convert_element_type_3, convert_element_type_2);  convert_element_type_3 = convert_element_type_2 = None
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul, [-1], True)
        fma: "f32[8192, 262144]" = torch.ops.prims.fma.default(neg, sum_1, mul);  neg = sum_1 = mul = None
        convert_element_type_4: "bf16[8192, 262144]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        return convert_element_type_4



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
