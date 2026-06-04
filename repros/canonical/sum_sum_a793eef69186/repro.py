"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_train_001
Pattern hash: a793eef69186
Shape hash: b3213d5b
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 464, 7, 7], f32), T([512, 232, 7, 7], f32), T([1, 232, 1, 1], f32), T([1, 232, 1, 1], f32), T([232], f32), T([232], f32), T([], f32), S([512, 232, 2, 7, 7]), S([512, 464, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[512, 464, 7, 7]", arg314_1: "f32[512, 232, 7, 7]", arg315_1: "f32[1, 232, 1, 1]", arg316_1: "f32[1, 232, 1, 1]", arg129_1: "f32[232]", arg130_1: "f32[232]", full: "f32[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[512, 232, 2, 7, 7]" = torch.ops.aten.view.default(getitem, _shape_param_0);  getitem = _shape_param_0 = None
        permute_default: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3, 4]);  view_default = None
        clone_default: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1: "f32[512, 464, 7, 7]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        slice_tensor: "f32[512, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_default_1, 1, 232, 464);  view_default_1 = None
        sub_tensor: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(arg314_1, arg315_1)
        mul_tensor: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, arg316_1);  sub_tensor = None
        unsqueeze_default: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1)
        unsqueeze_default_1: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_default_3: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[512, 232, 7, 7]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[512, 232, 7, 7]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[512, 232, 7, 7]" = torch.ops.aten.where.self(le_scalar, full, slice_tensor);  le_scalar = full = slice_tensor = None
        squeeze_dims: "f32[232]" = torch.ops.aten.squeeze.dims(arg315_1, [0, 2, 3]);  arg315_1 = None
        unsqueeze_default_4: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(arg314_1, unsqueeze_default_6);  arg314_1 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[232]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.985969387755102e-05);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[232]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.985969387755102e-05)
        squeeze_dims_1: "f32[232]" = torch.ops.aten.squeeze.dims(arg316_1, [0, 2, 3]);  arg316_1 = None
        mul_tensor_5: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[232]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg129_1);  arg129_1 = None
        unsqueeze_default_13: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[232]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        return (mul_tensor_9, mul_tensor_10)

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
