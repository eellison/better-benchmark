"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_train
Pattern hash: 874ba88ee22e
Shape hash: 44cc7dfb
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
    def forward(self, arg0_1: "bf16[128, 64, 8, 8]", arg1_1: "bf16[128, 64, 8, 8]", arg2_1: "b8[128, 64, 8, 8]", arg3_1: "f32[]", arg4_1: "bf16[128, 64, 8, 8]", arg5_1: "f32[64]", arg6_1: "f32[128, 32]", arg7_1: "f32[128, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10):
        # No stacktrace found for following nodes
        convert_element_type: "f32[128, 64, 8, 8]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float32);  arg0_1 = None
        convert_element_type_1: "f32[128, 64, 8, 8]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        add: "f32[128, 64, 8, 8]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        where: "f32[128, 64, 8, 8]" = torch.ops.aten.where.self(arg2_1, arg3_1, add);  arg2_1 = arg3_1 = add = None
        convert_element_type_2: "f32[128, 64, 8, 8]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        mul: "f32[128, 64, 8, 8]" = torch.ops.aten.mul.Tensor(where, convert_element_type_2)
        view: "f32[128, 64, 64]" = torch.ops.aten.view.default(mul, _shape_param_0);  mul = _shape_param_0 = None
        sum_1: "f32[128, 64]" = torch.ops.aten.sum.dim_IntList(view, [2]);  view = None
        view_1: "f32[128, 64, 64]" = torch.ops.aten.view.default(where, _shape_param_1);  _shape_param_1 = None
        sum_2: "f32[128, 64]" = torch.ops.aten.sum.dim_IntList(view_1, [2]);  view_1 = None
        unsqueeze: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg5_1, 0)
        mul_1: "f32[128, 64]" = torch.ops.aten.mul.Tensor(sum_1, unsqueeze)
        view_2: "f32[128, 32, 2]" = torch.ops.aten.view.default(mul_1, _shape_param_2);  mul_1 = _shape_param_2 = None
        sum_3: "f32[128, 32]" = torch.ops.aten.sum.dim_IntList(view_2, [2]);  view_2 = None
        mul_2: "f32[128, 64]" = torch.ops.aten.mul.Tensor(sum_2, unsqueeze);  unsqueeze = None
        view_3: "f32[128, 32, 2]" = torch.ops.aten.view.default(mul_2, _shape_param_3);  mul_2 = _shape_param_3 = None
        sum_4: "f32[128, 32]" = torch.ops.aten.sum.dim_IntList(view_3, [2]);  view_3 = None
        unsqueeze_1: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1)
        view_4: "f32[1, 32, 2]" = torch.ops.aten.view.default(arg5_1, _shape_param_4);  arg5_1 = _shape_param_4 = None
        mul_3: "f32[128, 32, 2]" = torch.ops.aten.mul.Tensor(unsqueeze_1, view_4);  view_4 = None
        mul_4: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sum_4, arg7_1)
        sub: "f32[128, 32]" = torch.ops.aten.sub.Tensor(mul_4, sum_3);  mul_4 = sum_3 = None
        mul_5: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sub, arg6_1);  sub = None
        mul_6: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_5, arg6_1);  mul_5 = None
        mul_7: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_6, arg6_1);  mul_6 = None
        mul_8: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_7, 0.0078125);  mul_7 = None
        neg: "f32[128, 32]" = torch.ops.aten.neg.default(mul_8)
        mul_9: "f32[128, 32]" = torch.ops.aten.mul.Tensor(neg, arg7_1);  neg = None
        mul_10: "f32[128, 32]" = torch.ops.aten.mul.Tensor(sum_4, arg6_1);  sum_4 = arg6_1 = None
        mul_11: "f32[128, 32]" = torch.ops.aten.mul.Tensor(mul_10, 0.0078125);  mul_10 = None
        sub_1: "f32[128, 32]" = torch.ops.aten.sub.Tensor(mul_9, mul_11);  mul_9 = mul_11 = None
        unsqueeze_2: "f32[128, 32, 2, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_3: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_8, -1);  mul_8 = None
        unsqueeze_4: "f32[128, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, -1);  unsqueeze_3 = None
        unsqueeze_5: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_1, -1);  sub_1 = None
        unsqueeze_6: "f32[128, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, -1);  unsqueeze_5 = None
        view_5: "f32[128, 32, 2, 64]" = torch.ops.aten.view.default(where, _shape_param_5);  _shape_param_5 = None
        mul_12: "f32[128, 32, 2, 64]" = torch.ops.aten.mul.Tensor(view_5, unsqueeze_2);  view_5 = unsqueeze_2 = None
        view_6: "f32[128, 32, 2, 64]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_6);  convert_element_type_2 = _shape_param_6 = None
        mul_13: "f32[128, 32, 2, 64]" = torch.ops.aten.mul.Tensor(view_6, unsqueeze_4);  view_6 = unsqueeze_4 = None
        add_1: "f32[128, 32, 2, 64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        add_2: "f32[128, 32, 2, 64]" = torch.ops.aten.add.Tensor(add_1, unsqueeze_6);  add_1 = unsqueeze_6 = None
        view_7: "f32[128, 64, 8, 8]" = torch.ops.aten.view.default(add_2, _shape_param_7);  add_2 = _shape_param_7 = None
        view_8: "f32[128, 32, 2]" = torch.ops.aten.view.default(sum_1, _shape_param_8);  sum_1 = _shape_param_8 = None
        view_9: "f32[128, 32, 2]" = torch.ops.aten.view.default(sum_2, _shape_param_9);  _shape_param_9 = None
        unsqueeze_7: "f32[128, 32, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        mul_14: "f32[128, 32, 2]" = torch.ops.aten.mul.Tensor(view_9, unsqueeze_7);  view_9 = unsqueeze_7 = None
        sub_2: "f32[128, 32, 2]" = torch.ops.aten.sub.Tensor(view_8, mul_14);  view_8 = mul_14 = None
        mul_15: "f32[128, 32, 2]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_1);  sub_2 = unsqueeze_1 = None
        sum_5: "f32[32, 2]" = torch.ops.aten.sum.dim_IntList(mul_15, [0]);  mul_15 = None
        view_10: "f32[64]" = torch.ops.aten.view.default(sum_5, _shape_param_10);  sum_5 = _shape_param_10 = None
        sum_6: "f32[64]" = torch.ops.aten.sum.dim_IntList(sum_2, [0]);  sum_2 = None
        convert_element_type_3: "bf16[128, 64, 8, 8]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        return (where, view_10, sum_6, convert_element_type_3)



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
