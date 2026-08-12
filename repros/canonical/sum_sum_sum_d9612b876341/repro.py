"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_train
Pattern hash: d9612b876341
Shape hash: 930536e2
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
    def forward(self, arg0_1: "f32[16384, 768]", arg1_1: "f32[768]", arg2_1: "f32[16384, 768]", arg3_1: "f32[32, 512, 1]", arg4_1: "f32[32, 512, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "f32[32, 512, 768]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, arg1_1);  arg1_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        view_1: "f32[32, 512, 768]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view_1, 0.5)
        pow_1: "f32[32, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_1, 3.0)
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(view_1, mul_3);  mul_3 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[32, 512, 768]" = torch.ops.aten.tanh.default(mul_4);  mul_4 = None
        add_1: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, add_1)
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_5, arg3_1);  mul_5 = arg3_1 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, arg4_1);  sub = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, mul_6);  mul = None
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [2], True);  mul_7 = None
        mul_8: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_6, sum_2);  sum_2 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_2: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_1, mul_8);  sub_1 = mul_8 = None
        div: "f32[32, 512, 1]" = torch.ops.aten.div.Tensor(arg4_1, 768);  arg4_1 = None
        mul_9: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(div, sub_2);  div = sub_2 = None
        mul_10: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(view, mul_6);  mul_6 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_10, [0, 1]);  mul_10 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(view, [0, 1]);  view = None
        mul_11: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, mul_2);  mul_2 = None
        mul_12: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, add_1);  mul_9 = add_1 = None
        mul_13: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(1, mul_13);  mul_13 = None
        mul_14: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_11, sub_3);  mul_11 = sub_3 = None
        mul_15: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_14, 0.7978845608028654);  mul_14 = None
        mul_16: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, 0.044715)
        pow_2: "f32[32, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_1, 2.0);  view_1 = None
        mul_17: "f32[32, 512, 768]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_18: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_15, mul_18);  mul_15 = mul_18 = None
        mul_19: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_12, 0.5);  mul_12 = None
        add_3: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_2, mul_19);  add_2 = mul_19 = None
        view_2: "f32[16384, 768]" = torch.ops.aten.view.default(add_3, _shape_param_2);  add_3 = _shape_param_2 = None
        permute: "f32[768, 16384]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True)
        view_3: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_3);  sum_5 = _shape_param_3 = None
        return (sum_3, sum_4, view_2, permute, view_3)



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
