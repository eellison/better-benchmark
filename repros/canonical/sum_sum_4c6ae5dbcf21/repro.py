"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_train
Pattern hash: 4c6ae5dbcf21
Shape hash: 1c935b51
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
    def forward(self, arg0_1: "bf16[4096, 512]", arg1_1: "f32[512]", arg2_1: "f32[32, 128, 512]", arg3_1: "f32[32, 128, 1]", arg4_1: "f32[32, 128, 512]", arg5_1: "b8[32, 128, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[32, 128, 512]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, arg1_1);  arg1_1 = None
        mul_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg2_1, arg3_1)
        mul_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type, mul_1);  convert_element_type = mul_1 = None
        sum_1: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_2, [0, 1], True, dtype = torch.float32);  mul_2 = None
        view_1: "f32[512]" = torch.ops.aten.view.default(sum_1, _shape_param_1);  sum_1 = _shape_param_1 = None
        mul_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul, arg2_1)
        mul_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = None
        sum_2: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [2], True, dtype = torch.float32);  mul_3 = None
        add: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(arg4_1, mul_4);  arg4_1 = mul_4 = None
        pow_1: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg3_1, 3);  arg3_1 = None
        mul_5: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_2, -0.5);  sum_2 = None
        mul_6: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_5, pow_1);  mul_5 = pow_1 = None
        expand: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_6, _shape_param_2);  mul_6 = _shape_param_2 = None
        div: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand, 512);  expand = None
        pow_2: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg2_1, 1.0);  arg2_1 = None
        mul_7: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        mul_8: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div, mul_7);  div = mul_7 = None
        add_1: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add, mul_8);  add = mul_8 = None
        convert_element_type_1: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16)
        convert_element_type_2: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        mul_9: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_10: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1, mul_9);  convert_element_type_1 = mul_9 = None
        view_2: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_10, _shape_param_3);  mul_10 = _shape_param_3 = None
        permute: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        return (view_1, add_1, view_2, permute)



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
