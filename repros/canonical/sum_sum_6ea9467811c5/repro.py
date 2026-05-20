"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-2-5-linux.aws.a100_graph5
Pattern hash: 6ea9467811c5
Shape hash: e910b387
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 768], f32), T([768], f32), T([512, 768], f32), T([1, 512, 1], f32), T([1, 512, 1], f32), T([768, 768], f32), S([1, 512, 768]), S([1, 512, 768]), S([512, 768]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[512, 768]", arg55_1: "f32[768]", arg133_1: "f32[512, 768]", arg134_1: "f32[1, 512, 1]", arg135_1: "f32[1, 512, 1]", arg54_1: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_default, arg55_1);  view_default = arg55_1 = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        view_default_1: "f32[1, 512, 768]" = torch.ops.aten.view.default(arg133_1, _shape_param_1);  arg133_1 = _shape_param_1 = None
        mul_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_default_1, 0.5)
        pow_tensor_scalar: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_default_1, 3.0)
        mul_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(view_default_1, mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default: "f32[1, 512, 768]" = torch.ops.aten.tanh.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_5: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1)
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_5, arg134_1);  mul_tensor_5 = arg134_1 = None
        mul_tensor_6: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, arg135_1);  sub_tensor = None
        mul_tensor_7: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_6);  mul_tensor = None
        sum_dim_int_list_1: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2], True);  mul_tensor_7 = None
        mul_tensor_8: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_6, sum_dim_int_list_1);  mul_tensor_6 = sum_dim_int_list_1 = None
        sub_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_8);  sub_tensor_1 = mul_tensor_8 = None
        div_tensor: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(arg135_1, 768);  arg135_1 = None
        mul_tensor_9: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_10: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_9, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_11: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_9, add_tensor_1);  mul_tensor_9 = add_tensor_1 = None
        mul_tensor_12: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(1, mul_tensor_12);  mul_tensor_12 = None
        mul_tensor_13: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_10, sub_tensor_3);  mul_tensor_10 = sub_tensor_3 = None
        mul_tensor_14: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_13, 0.7978845608028654);  mul_tensor_13 = None
        mul_tensor_15: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_14, 0.044715)
        pow_tensor_scalar_1: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(view_default_1, 2.0);  view_default_1 = None
        mul_scalar: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_16: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_15, mul_scalar);  mul_tensor_15 = mul_scalar = None
        add_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_14, mul_tensor_16);  mul_tensor_14 = mul_tensor_16 = None
        mul_tensor_17: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_11, 0.5);  mul_tensor_11 = None
        add_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_17);  add_tensor_2 = mul_tensor_17 = None
        view_default_2: "f32[512, 768]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_2);  add_tensor_3 = _shape_param_2 = None
        permute_default: "f32[768, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (view_default_2, permute_default_1)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
