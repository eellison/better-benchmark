"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train
Pattern hash: be519a2435f2
Shape hash: 0d0b1d4e
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
    def forward(self, arg0_1: "f32[16384, 768]", arg1_1: "f32[768]", arg2_1: "f32[32, 512, 768]", arg3_1: "f32[32, 512, 1]", arg4_1: "b8[32, 512, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "f32[32, 512, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg1_1);  arg1_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = None
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg2_1, sum_2);  sum_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg3_1, sub_1);  arg3_1 = sub_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg2_1);  arg2_1 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(view, [0, 1]);  view = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, mul_6);  mul_6 = None
        view_1: "f32[16384, 768]" = torch.ops.aten.view.default(mul_7, _shape_param_1);  mul_7 = _shape_param_1 = None
        permute: "f32[768, 16384]" = torch.ops.aten.permute.default(view_1, [1, 0])
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True)
        view_2: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_2);  sum_5 = _shape_param_2 = None
        return (mul_4, sum_3, sum_4, view_1, permute, view_2)



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
