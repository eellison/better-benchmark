"""
Standalone repro captured via capture_hook.
Label: GoogleFnet
Pattern hash: d997bc00cece
Shape hash: 0fcc6126
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
    def forward(self, arg0_1: "f32[32, 512, 768, 2]", arg1_1: "f32[32, 512, 768]", arg2_1: "f32[768]", arg3_1: "f32[32, 512, 768]", arg4_1: "f32[32, 512, 1]", arg5_1: "b8[32, 512, 768]", arg6_1: "f32[768, 3072]", arg7_1: "f32[16384, 3072]", arg8_1: "f32[16384, 3072]", arg9_1: "f32[3072, 768]", arg10_1: "f32[16384, 768]", arg11_1: "f32[768]", arg12_1: "f32[32, 512, 768]", arg13_1: "f32[32, 512, 1]", arg14_1: "f32[32, 512, 768, 2]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        select: "f32[32, 512, 768]" = torch.ops.aten.select.int(arg0_1, 3, 0);  arg0_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg1_1, select);  arg1_1 = select = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add, arg2_1);  arg2_1 = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, 768)
        sum_1: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul, [2], True)
        mul_2: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg3_1);  mul = None
        sum_2: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True);  mul_2 = None
        mul_3: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg3_1, sum_2);  sum_2 = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_1, sum_1);  mul_1 = sum_1 = None
        sub_1: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub, mul_3);  sub = mul_3 = None
        mul_4: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg4_1, sub_1);  arg4_1 = sub_1 = None
        mul_5: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add, arg3_1);  arg3_1 = None
        sum_3: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_5, [0, 1]);  mul_5 = None
        sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(add, [0, 1]);  add = None
        convert_element_type: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float32);  arg5_1 = None
        mul_6: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_7: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, mul_6);  mul_6 = None
        view: "f32[16384, 768]" = torch.ops.aten.view.default(mul_7, _shape_param_0);  mul_7 = _shape_param_0 = None
        permute: "f32[3072, 768]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        permute_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[16384, 3072]" = torch.ops.aten.mm.default(view, permute_1);  permute_1 = None
        permute_2: "f32[768, 16384]" = torch.ops.aten.permute.default(view, [1, 0])
        mm_1: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_2, arg7_1);  permute_2 = arg7_1 = None
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view, [0], True);  view = None
        view_1: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_1);  sum_5 = _shape_param_1 = None
        view_2: "f32[32, 512, 3072]" = torch.ops.aten.view.default(mm, _shape_param_2);  mm = _shape_param_2 = None
        view_3: "f32[32, 512, 3072]" = torch.ops.aten.view.default(arg8_1, _shape_param_3);  arg8_1 = _shape_param_3 = None
        mul_8: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_3, 0.5)
        mul_9: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_2, mul_8);  mul_8 = None
        pow_1: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_3, 3.0)
        mul_10: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_1: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(view_3, mul_10);  mul_10 = None
        mul_11: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_1, 0.7978845608028654);  add_1 = None
        tanh: "f32[32, 512, 3072]" = torch.ops.aten.tanh.default(mul_11);  mul_11 = None
        add_2: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_12: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(view_2, add_2);  view_2 = add_2 = None
        mul_13: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_2: "f32[32, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_13);  mul_13 = None
        mul_14: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_9, sub_2);  mul_9 = sub_2 = None
        mul_15: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_14, 0.7978845608028654);  mul_14 = None
        mul_16: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_15, 0.044715)
        pow_2: "f32[32, 512, 3072]" = torch.ops.aten.pow.Tensor_Scalar(view_3, 2.0);  view_3 = None
        mul_17: "f32[32, 512, 3072]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_18: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        add_3: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_15, mul_18);  mul_15 = mul_18 = None
        mul_19: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_12, 0.5);  mul_12 = None
        add_4: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(add_3, mul_19);  add_3 = mul_19 = None
        view_4: "f32[16384, 3072]" = torch.ops.aten.view.default(add_4, _shape_param_4);  add_4 = _shape_param_4 = None
        permute_3: "f32[768, 3072]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        permute_4: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mm_2: "f32[16384, 768]" = torch.ops.aten.mm.default(view_4, permute_4);  permute_4 = None
        permute_5: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_3: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_5, arg10_1);  permute_5 = arg10_1 = None
        sum_6: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True);  view_4 = None
        view_5: "f32[3072]" = torch.ops.aten.view.default(sum_6, _shape_param_5);  sum_6 = _shape_param_5 = None
        view_6: "f32[32, 512, 768]" = torch.ops.aten.view.default(mm_2, _shape_param_6);  mm_2 = _shape_param_6 = None
        add_5: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_4, view_6);  mul_4 = view_6 = None
        mul_20: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_5, arg11_1);  arg11_1 = None
        mul_21: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, 768)
        sum_7: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True)
        mul_22: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg12_1);  mul_20 = None
        sum_8: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [2], True);  mul_22 = None
        mul_23: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg12_1, sum_8);  sum_8 = None
        sub_3: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_21, sum_7);  mul_21 = sum_7 = None
        sub_4: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_3, mul_23);  sub_3 = mul_23 = None
        mul_24: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg13_1, sub_4);  arg13_1 = sub_4 = None
        mul_25: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_5, arg12_1);  arg12_1 = None
        sum_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_25, [0, 1]);  mul_25 = None
        sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_5, [0, 1]);  add_5 = None
        select_scatter: "f32[32, 512, 768, 2]" = torch.ops.aten.select_scatter.default(arg14_1, mul_24, 3, 0);  arg14_1 = None
        return (sum_3, sum_4, mm_1, view_1, mm_3, view_5, mul_24, sum_9, sum_10, select_scatter)



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
