"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: 34c857ab7db3
Shape hash: 574395aa
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), S([16384]), S([16384]), S([16384]), S([16384]), S([16384]), S([16384]), S([16384]), S([16384]), S([16384]), S([16384]), S([16384]), S([8, 512, 16384]), S([8, 512, 16384]), S([4096, 16384]), S([16384]))"

class Repro(torch.nn.Module):
    def forward(self, view_15: "f32[4096, 16384]", view_45: "f32[4096, 16384]", view_75: "f32[4096, 16384]", view_105: "f32[4096, 16384]", view_135: "f32[4096, 16384]", view_165: "f32[4096, 16384]", view_195: "f32[4096, 16384]", view_225: "f32[4096, 16384]", view_255: "f32[4096, 16384]", view_285: "f32[4096, 16384]", view_315: "f32[4096, 16384]", mm_136: "f32[4096, 16384]", arg24_1: "f32[4096, 16384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_15, [0], True);  view_15 = None
        view_default: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        sum_dim_int_list_1: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_45, [0], True);  view_45 = None
        view_default_1: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        add_tensor: "f32[16384]" = torch.ops.aten.add.Tensor(view_default, view_default_1);  view_default = view_default_1 = None
        sum_dim_int_list_2: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True);  view_75 = None
        view_default_2: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list_2, _shape_param_2);  sum_dim_int_list_2 = _shape_param_2 = None
        add_tensor_1: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor, view_default_2);  add_tensor = view_default_2 = None
        sum_dim_int_list_3: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True);  view_105 = None
        view_default_3: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list_3, _shape_param_3);  sum_dim_int_list_3 = _shape_param_3 = None
        add_tensor_2: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_1, view_default_3);  add_tensor_1 = view_default_3 = None
        sum_dim_int_list_4: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
        view_default_4: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_4);  sum_dim_int_list_4 = _shape_param_4 = None
        add_tensor_3: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_2, view_default_4);  add_tensor_2 = view_default_4 = None
        sum_dim_int_list_5: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
        view_default_5: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list_5, _shape_param_5);  sum_dim_int_list_5 = _shape_param_5 = None
        add_tensor_4: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_3, view_default_5);  add_tensor_3 = view_default_5 = None
        sum_dim_int_list_6: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True);  view_195 = None
        view_default_6: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list_6, _shape_param_6);  sum_dim_int_list_6 = _shape_param_6 = None
        add_tensor_5: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_4, view_default_6);  add_tensor_4 = view_default_6 = None
        sum_dim_int_list_7: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_225, [0], True);  view_225 = None
        view_default_7: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list_7, _shape_param_7);  sum_dim_int_list_7 = _shape_param_7 = None
        add_tensor_6: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_5, view_default_7);  add_tensor_5 = view_default_7 = None
        sum_dim_int_list_8: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_255, [0], True);  view_255 = None
        view_default_8: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list_8, _shape_param_8);  sum_dim_int_list_8 = _shape_param_8 = None
        add_tensor_7: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_6, view_default_8);  add_tensor_6 = view_default_8 = None
        sum_dim_int_list_9: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
        view_default_9: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list_9, _shape_param_9);  sum_dim_int_list_9 = _shape_param_9 = None
        add_tensor_8: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_7, view_default_9);  add_tensor_7 = view_default_9 = None
        sum_dim_int_list_10: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
        view_default_10: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list_10, _shape_param_10);  sum_dim_int_list_10 = _shape_param_10 = None
        add_tensor_9: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_8, view_default_10);  add_tensor_8 = view_default_10 = None
        view_default_11: "f32[8, 512, 16384]" = torch.ops.aten.view.default(mm_136, _shape_param_11);  mm_136 = _shape_param_11 = None
        view_default_12: "f32[8, 512, 16384]" = torch.ops.aten.view.default(arg24_1, _shape_param_12);  arg24_1 = _shape_param_12 = None
        mul_tensor: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_default_12, 0.5)
        mul_tensor_1: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_default_11, mul_tensor);  mul_tensor = None
        pow_tensor_scalar: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_default_12, 3.0)
        mul_tensor_2: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_10: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(view_default_12, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add_tensor_10, 0.7978845608028654);  add_tensor_10 = None
        tanh_default: "f32[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_11: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh_default, 1.0)
        mul_tensor_4: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view_default_11, add_tensor_11);  view_default_11 = add_tensor_11 = None
        mul_tensor_5: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor: "f32[8, 512, 16384]" = torch.ops.aten.sub.Tensor(1, mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor);  mul_tensor_1 = sub_tensor = None
        mul_tensor_7: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_6, 0.7978845608028654);  mul_tensor_6 = None
        mul_tensor_8: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.044715)
        pow_tensor_scalar_1: "f32[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_default_12, 2.0);  view_default_12 = None
        mul_scalar: "f32[8, 512, 16384]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_9: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_scalar);  mul_tensor_8 = mul_scalar = None
        add_tensor_12: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = mul_tensor_9 = None
        mul_tensor_10: "f32[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.5);  mul_tensor_4 = None
        add_tensor_13: "f32[8, 512, 16384]" = torch.ops.aten.add.Tensor(add_tensor_12, mul_tensor_10);  add_tensor_12 = mul_tensor_10 = None
        view_default_13: "f32[4096, 16384]" = torch.ops.aten.view.default(add_tensor_13, _shape_param_13);  add_tensor_13 = _shape_param_13 = None
        permute_default: "f32[16384, 4096]" = torch.ops.aten.permute.default(view_default_13, [1, 0])
        sum_dim_int_list_11: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_default_13, [0], True);  view_default_13 = None
        view_default_14: "f32[16384]" = torch.ops.aten.view.default(sum_dim_int_list_11, _shape_param_14);  sum_dim_int_list_11 = _shape_param_14 = None
        add_tensor_14: "f32[16384]" = torch.ops.aten.add.Tensor(add_tensor_9, view_default_14);  add_tensor_9 = view_default_14 = None
        return (permute_default, add_tensor_14)

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
