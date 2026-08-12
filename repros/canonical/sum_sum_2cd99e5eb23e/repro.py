"""
Standalone repro captured via capture_hook.
Label: genai_RMSNormBackward_static
Pattern hash: 2cd99e5eb23e
Shape hash: cbc6f48e
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
    def forward(self, arg0_1: "bf16[]", arg1_1: "f32[512]", arg2_1: "f32[1152000, 1]", arg3_1: "bf16[1152000, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        expand: "bf16[1152000, 512]" = torch.ops.aten.expand.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(expand, torch.float32);  expand = None
        mul: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, arg1_1);  arg1_1 = None
        mul_1: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul, arg2_1)
        convert_element_type_1: "f32[1152000, 512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        mul_2: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(mul, convert_element_type_1);  mul = None
        sum_1: "f32[1152000, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [1], True);  mul_2 = None
        mul_3: "f32[1152000, 1]" = torch.ops.aten.mul.Scalar(sum_1, -0.5);  sum_1 = None
        pow_1: "f32[1152000, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg2_1, 3)
        mul_4: "f32[1152000, 1]" = torch.ops.aten.mul.Tensor(mul_3, pow_1);  mul_3 = pow_1 = None
        expand_1: "f32[1152000, 512]" = torch.ops.aten.expand.default(mul_4, _shape_param_1);  mul_4 = _shape_param_1 = None
        div: "f32[1152000, 512]" = torch.ops.aten.div.Scalar(expand_1, 512);  expand_1 = None
        pow_2: "f32[1152000, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 1.0)
        mul_5: "f32[1152000, 512]" = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        mul_6: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(div, mul_5);  div = mul_5 = None
        add: "f32[1152000, 512]" = torch.ops.aten.add.Tensor(mul_1, mul_6);  mul_1 = mul_6 = None
        convert_element_type_2: "bf16[1152000, 512]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        mul_7: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, arg2_1);  convert_element_type_1 = arg2_1 = None
        mul_8: "f32[1152000, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_7);  convert_element_type = mul_7 = None
        sum_2: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(mul_8, [0], True);  mul_8 = None
        view: "f32[512]" = torch.ops.aten.view.default(sum_2, _shape_param_2);  sum_2 = _shape_param_2 = None
        return (convert_element_type_2, view)



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
