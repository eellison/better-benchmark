"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_train_001
Pattern hash: 8faedc560651
Shape hash: bd904f60
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 512], f32), T([64, 512, 1, 1], b8), T([64, 512, 1, 1], f32), T([512], f32), T([64, 32], f32), T([64, 32], f32), S([64, 512, 1, 1]), S([64, 512, 1, 1]), S([64, 512, 1]), S([64, 512, 1]), S([64, 32, 16]), S([64, 32, 16]), S([1, 32, 16]), S([64, 32, 16, 1]), S([64, 32, 16, 1]), S([64, 512, 1, 1]), S([64, 32, 16]), S([64, 32, 16]), S([512]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[64, 512]", arg121_1: "b8[64, 512, 1, 1]", arg117_1: "f32[64, 512, 1, 1]", arg41_1: "f32[512]", arg119_1: "f32[64, 32]", arg118_1: "f32[64, 32]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # No stacktrace found for following nodes
        view_default: "f32[64, 512, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        expand_default: "f32[64, 512, 1, 1]" = torch.ops.aten.expand.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        div_scalar: "f32[64, 512, 1, 1]" = torch.ops.aten.div.Scalar(expand_default, 1);  expand_default = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[64, 512, 1, 1]" = torch.ops.aten.where.self(arg121_1, full_default, div_scalar);  arg121_1 = full_default = div_scalar = None
        mul_tensor: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(where_self, arg117_1)
        view_default_1: "f32[64, 512, 1]" = torch.ops.aten.view.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
        sum_dim_int_list: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(view_default_1, [2]);  view_default_1 = None
        view_default_2: "f32[64, 512, 1]" = torch.ops.aten.view.default(where_self, _shape_param_3);  _shape_param_3 = None
        sum_dim_int_list_1: "f32[64, 512]" = torch.ops.aten.sum.dim_IntList(view_default_2, [2]);  view_default_2 = None
        unsqueeze_default: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(arg41_1, 0)
        mul_tensor_1: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, unsqueeze_default)
        view_default_3: "f32[64, 32, 16]" = torch.ops.aten.view.default(mul_tensor_1, _shape_param_4);  mul_tensor_1 = _shape_param_4 = None
        sum_dim_int_list_2: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_default_3, [2]);  view_default_3 = None
        mul_tensor_2: "f32[64, 512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, unsqueeze_default);  unsqueeze_default = None
        view_default_4: "f32[64, 32, 16]" = torch.ops.aten.view.default(mul_tensor_2, _shape_param_5);  mul_tensor_2 = _shape_param_5 = None
        sum_dim_int_list_3: "f32[64, 32]" = torch.ops.aten.sum.dim_IntList(view_default_4, [2]);  view_default_4 = None
        unsqueeze_default_1: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1)
        view_default_5: "f32[1, 32, 16]" = torch.ops.aten.view.default(arg41_1, _shape_param_6);  arg41_1 = _shape_param_6 = None
        mul_tensor_3: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(unsqueeze_default_1, view_default_5);  view_default_5 = None
        mul_tensor_4: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, arg118_1)
        sub_tensor: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list_2);  mul_tensor_4 = sum_dim_int_list_2 = None
        mul_tensor_5: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sub_tensor, arg119_1);  sub_tensor = None
        mul_tensor_6: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_5, arg119_1);  mul_tensor_5 = None
        mul_tensor_7: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_6, arg119_1);  mul_tensor_6 = None
        mul_tensor_8: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.0625);  mul_tensor_7 = None
        neg_default: "f32[64, 32]" = torch.ops.aten.neg.default(mul_tensor_8)
        mul_tensor_9: "f32[64, 32]" = torch.ops.aten.mul.Tensor(neg_default, arg118_1);  neg_default = None
        mul_tensor_10: "f32[64, 32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, arg119_1);  sum_dim_int_list_3 = arg119_1 = None
        mul_tensor_11: "f32[64, 32]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.0625);  mul_tensor_10 = None
        sub_tensor_1: "f32[64, 32]" = torch.ops.aten.sub.Tensor(mul_tensor_9, mul_tensor_11);  mul_tensor_9 = mul_tensor_11 = None
        unsqueeze_default_2: "f32[64, 32, 16, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_3: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_8, -1);  mul_tensor_8 = None
        unsqueeze_default_4: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, -1);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(sub_tensor_1, -1);  sub_tensor_1 = None
        unsqueeze_default_6: "f32[64, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, -1);  unsqueeze_default_5 = None
        view_default_6: "f32[64, 32, 16, 1]" = torch.ops.aten.view.default(where_self, _shape_param_7);  where_self = _shape_param_7 = None
        mul_tensor_12: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_default_6, unsqueeze_default_2);  view_default_6 = unsqueeze_default_2 = None
        view_default_7: "f32[64, 32, 16, 1]" = torch.ops.aten.view.default(arg117_1, _shape_param_8);  arg117_1 = _shape_param_8 = None
        mul_tensor_13: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(view_default_7, unsqueeze_default_4);  view_default_7 = unsqueeze_default_4 = None
        add_tensor: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(mul_tensor_12, mul_tensor_13);  mul_tensor_12 = mul_tensor_13 = None
        add_tensor_1: "f32[64, 32, 16, 1]" = torch.ops.aten.add.Tensor(add_tensor, unsqueeze_default_6);  add_tensor = unsqueeze_default_6 = None
        view_default_8: "f32[64, 512, 1, 1]" = torch.ops.aten.view.default(add_tensor_1, _shape_param_9);  add_tensor_1 = _shape_param_9 = None
        view_default_9: "f32[64, 32, 16]" = torch.ops.aten.view.default(sum_dim_int_list, _shape_param_10);  sum_dim_int_list = _shape_param_10 = None
        view_default_10: "f32[64, 32, 16]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_11);  _shape_param_11 = None
        unsqueeze_default_7: "f32[64, 32, 1]" = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        mul_tensor_14: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(view_default_10, unsqueeze_default_7);  view_default_10 = unsqueeze_default_7 = None
        sub_tensor_2: "f32[64, 32, 16]" = torch.ops.aten.sub.Tensor(view_default_9, mul_tensor_14);  view_default_9 = mul_tensor_14 = None
        mul_tensor_15: "f32[64, 32, 16]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_1);  sub_tensor_2 = unsqueeze_default_1 = None
        sum_dim_int_list_4: "f32[32, 16]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0]);  mul_tensor_15 = None
        view_default_11: "f32[512]" = torch.ops.aten.view.default(sum_dim_int_list_4, _shape_param_12);  sum_dim_int_list_4 = _shape_param_12 = None
        sum_dim_int_list_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(sum_dim_int_list_1, [0]);  sum_dim_int_list_1 = None
        return (view_default_8, view_default_11, sum_dim_int_list_5)



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
