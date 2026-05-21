"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_train_001
Pattern hash: 8b606a2377c7
Shape hash: c9d1356e
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
_shapes_config = "(T([32, 128, 28, 28], f32), T([32, 128, 56, 56], f32), T([32, 256, 56, 56], f32), T([1, 256, 1, 1], f32), T([1, 256, 1, 1], f32), T([256], f32), T([256], f32), T([32, 256, 1, 1], f32), S([32, 2, 128, 56, 56]), S([32, 2, 128, 56, 56]), S([32, 1, 2, -1]), S([32, 256, 1, 1]), S([32, 256]), S([32, 2, 1, 128]), S([32, 256, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_39: "f32[32, 128, 28, 28]", arg98_1: "f32[32, 128, 56, 56]", arg90_1: "f32[32, 256, 56, 56]", arg91_1: "f32[1, 256, 1, 1]", arg92_1: "f32[1, 256, 1, 1]", arg23_1: "f32[256]", arg24_1: "f32[256]", arg97_1: "f32[32, 256, 1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        avg_pool2d_backward_default: "f32[32, 128, 56, 56]" = torch.ops.aten.avg_pool2d_backward.default(getitem_39, arg98_1, [3, 3], [2, 2], [1, 1], False, True, None);  getitem_39 = arg98_1 = None
        unsqueeze_default: "f32[32, 1, 128, 56, 56]" = torch.ops.aten.unsqueeze.default(avg_pool2d_backward_default, 1);  avg_pool2d_backward_default = None
        expand_default: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        sub_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(arg90_1, arg91_1);  arg90_1 = arg91_1 = None
        mul_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, arg92_1);  sub_tensor = arg92_1 = None
        unsqueeze_default_1: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_default_2: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, -1);  unsqueeze_default_1 = None
        mul_tensor_1: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_2);  mul_tensor = unsqueeze_default_2 = None
        unsqueeze_default_3: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_default_4: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, -1);  unsqueeze_default_3 = None
        add_tensor: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_4);  mul_tensor_1 = unsqueeze_default_4 = None
        relu_default: "f32[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        view_default: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.view.default(relu_default, _shape_param_1);  relu_default = _shape_param_1 = None
        mul_tensor_2: "f32[32, 2, 128, 56, 56]" = torch.ops.aten.mul.Tensor(expand_default, view_default);  expand_default = view_default = None
        view_default_1: "f32[32, 1, 2, 128]" = torch.ops.aten.view.default(arg97_1, _shape_param_2);  arg97_1 = _shape_param_2 = None
        permute_default: "f32[32, 2, 1, 128]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        amax_default: "f32[32, 1, 1, 128]" = torch.ops.aten.amax.default(permute_default, [1], True)
        sub_tensor_1: "f32[32, 2, 1, 128]" = torch.ops.aten.sub.Tensor(permute_default, amax_default);  permute_default = amax_default = None
        exp_default: "f32[32, 2, 1, 128]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[32, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True)
        div_tensor: "f32[32, 2, 1, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sum_dim_int_list_1: "f32[32, 2, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3, 4], True);  mul_tensor_2 = None
        view_default_2: "f32[32, 256, 1, 1]" = torch.ops.aten.view.default(sum_dim_int_list_1, _shape_param_3);  sum_dim_int_list_1 = _shape_param_3 = None
        view_default_3: "f32[32, 256]" = torch.ops.aten.view.default(view_default_2, _shape_param_4);  view_default_2 = _shape_param_4 = None
        view_default_4: "f32[32, 2, 1, 128]" = torch.ops.aten.view.default(view_default_3, _shape_param_5);  view_default_3 = _shape_param_5 = None
        mul_tensor_3: "f32[32, 2, 1, 128]" = torch.ops.aten.mul.Tensor(view_default_4, div_tensor);  view_default_4 = None
        sum_dim_int_list_2: "f32[32, 1, 1, 128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [1], True)
        neg_default: "f32[32, 2, 1, 128]" = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default: "f32[32, 2, 1, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list_2, mul_tensor_3);  neg_default = sum_dim_int_list_2 = mul_tensor_3 = None
        permute_default_1: "f32[32, 1, 2, 128]" = torch.ops.aten.permute.default(fma_default, [0, 2, 1, 3]);  fma_default = None
        view_default_5: "f32[32, 256, 1, 1]" = torch.ops.aten.view.default(permute_default_1, _shape_param_6);  permute_default_1 = _shape_param_6 = None
        sum_dim_int_list_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_default_5, [0, 2, 3]);  view_default_5 = None
        return sum_dim_int_list_3



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
