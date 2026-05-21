"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train_001
Pattern hash: a7c7ab7bfcf5
Shape hash: 5f2668c9
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
_shapes_config = "(T([128, 64, 73, 73], f32), T([128, 64, 73, 73], i8, gen=Index(9)), T([128, 64, 147, 147], f32), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([], f32), S([8192, 5329]), S([8192, 5329]), S([128, 64, 147, 147]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_270: "f32[128, 64, 73, 73]", arg246_1: "i8[128, 64, 73, 73]", arg242_1: "f32[128, 64, 147, 147]", arg243_1: "f32[1, 64, 1, 1]", arg244_1: "f32[1, 64, 1, 1]", arg6_1: "f32[64]", arg7_1: "f32[64]", full_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        full_default: "f32[8192, 21609]" = torch.ops.aten.full.default([8192, 21609], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        clone_default: "f32[128, 64, 73, 73]" = torch.ops.aten.clone.default(getitem_270, memory_format = torch.contiguous_format);  getitem_270 = None
        view_default: "f32[8192, 5329]" = torch.ops.aten.view.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[128, 64, 73, 73]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(arg246_1, [3, 3], [147, 147], [2, 2], [0, 0], [1, 1]);  arg246_1 = None
        clone_default_1: "i64[128, 64, 73, 73]" = torch.ops.aten.clone.default(_low_memory_max_pool_offsets_to_indices_default, memory_format = torch.contiguous_format);  _low_memory_max_pool_offsets_to_indices_default = None
        view_default_1: "i64[8192, 5329]" = torch.ops.aten.view.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        scatter_add_default: "f32[8192, 21609]" = torch.ops.aten.scatter_add.default(full_default, 1, view_default_1, view_default);  full_default = view_default_1 = view_default = None
        view_default_2: "f32[128, 64, 147, 147]" = torch.ops.aten.view.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None
        clone_default_2: "f32[128, 64, 147, 147]" = torch.ops.aten.clone.default(view_default_2, memory_format = torch.channels_last);  view_default_2 = None
        sub_tensor: "f32[128, 64, 147, 147]" = torch.ops.aten.sub.Tensor(arg242_1, arg243_1)
        mul_tensor: "f32[128, 64, 147, 147]" = torch.ops.aten.mul.Tensor(sub_tensor, arg244_1);  sub_tensor = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1)
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 64, 147, 147]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 64, 147, 147]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[128, 64, 147, 147]" = torch.ops.aten.relu.default(add_tensor);  add_tensor = None
        le_scalar: "b8[128, 64, 147, 147]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        where_self: "f32[128, 64, 147, 147]" = torch.ops.aten.where.self(le_scalar, full_1, clone_default_2);  le_scalar = full_1 = clone_default_2 = None
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(arg243_1, [0, 2, 3]);  arg243_1 = None
        unsqueeze_default_4: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
        sub_tensor_1: "f32[128, 64, 147, 147]" = torch.ops.aten.sub.Tensor(arg242_1, unsqueeze_default_6);  arg242_1 = unsqueeze_default_6 = None
        mul_tensor_2: "f32[128, 64, 147, 147]" = torch.ops.aten.mul.Tensor(where_self, sub_tensor_1)
        sum_dim_int_list_1: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3]);  mul_tensor_2 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 3.6153917349252627e-07);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_8: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 3.6153917349252627e-07)
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(arg244_1, [0, 2, 3]);  arg244_1 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_6: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_10: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_11: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_7: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg6_1);  arg6_1 = None
        unsqueeze_default_13: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_14: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_8: "f32[128, 64, 147, 147]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_12);  sub_tensor_1 = unsqueeze_default_12 = None
        sub_tensor_2: "f32[128, 64, 147, 147]" = torch.ops.aten.sub.Tensor(where_self, mul_tensor_8);  where_self = mul_tensor_8 = None
        sub_tensor_3: "f32[128, 64, 147, 147]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_9);  sub_tensor_2 = unsqueeze_default_9 = None
        mul_tensor_9: "f32[128, 64, 147, 147]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_15);  sub_tensor_3 = unsqueeze_default_15 = None
        mul_tensor_10: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
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
