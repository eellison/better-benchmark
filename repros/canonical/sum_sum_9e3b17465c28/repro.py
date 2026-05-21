"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train_001
Pattern hash: 9e3b17465c28
Shape hash: 6cf4583b
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
_shapes_config = "(T([128, 1280], f32), T([128, 1280, 7, 7], f32), T([1, 1280, 1, 1], f32), T([1, 1280, 1, 1], f32), T([1280], f32), T([1280], f32), S([128, 1280, 1, 1]), S([128, 1280, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[128, 1280]", arg405_1: "f32[128, 1280, 7, 7]", arg406_1: "f32[1, 1280, 1, 1]", arg407_1: "f32[1, 1280, 1, 1]", arg161_1: "f32[1280]", arg162_1: "f32[1280]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 1280, 1, 1]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        squeeze_dim: "f32[128, 1280, 1]" = torch.ops.aten.squeeze.dim(view_default, 3);  view_default = None
        squeeze_dim_1: "f32[128, 1280]" = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default: "f32[163840]" = torch.ops.aten.full.default([163840], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter_default: "f32[163840]" = torch.ops.aten.as_strided_scatter.default(full_default, squeeze_dim_1, [128, 1280], [1280, 1], 0);  full_default = squeeze_dim_1 = None
        as_strided_default: "f32[128, 1280, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter_default, [128, 1280, 1, 1], [1280, 1, 1, 1], 0);  as_strided_scatter_default = None
        expand_default: "f32[128, 1280, 7, 7]" = torch.ops.aten.expand.default(as_strided_default, _shape_param_1);  as_strided_default = _shape_param_1 = None
        div_scalar: "f32[128, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        sub_tensor: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(arg405_1, arg406_1)
        mul_tensor: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, arg407_1);  sub_tensor = None
        unsqueeze_default: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg161_1, -1)
        unsqueeze_default_1: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_default_3: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        neg_default: "f32[128, 1280, 7, 7]" = torch.ops.aten.neg.default(add_tensor)
        exp_default: "f32[128, 1280, 7, 7]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_1: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        reciprocal_default: "f32[128, 1280, 7, 7]" = torch.ops.aten.reciprocal.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_3: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(div_scalar, mul_tensor_2);  div_scalar = None
        sub_tensor_1: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_4: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor, sub_tensor_1);  add_tensor = sub_tensor_1 = None
        add_tensor_2: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_4, 1);  mul_tensor_4 = None
        mul_tensor_5: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor_2);  mul_tensor_3 = add_tensor_2 = None
        squeeze_dims: "f32[1280]" = torch.ops.aten.squeeze.dims(arg406_1, [0, 2, 3]);  arg406_1 = None
        unsqueeze_default_4: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        sum_dim_int_list: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 2, 3])
        sub_tensor_2: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(arg405_1, unsqueeze_default_6);  arg405_1 = unsqueeze_default_6 = None
        mul_tensor_6: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_5, sub_tensor_2)
        sum_dim_int_list_1: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 2, 3]);  mul_tensor_6 = None
        mul_tensor_7: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00015943877551020407);  sum_dim_int_list = None
        unsqueeze_default_7: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_7, 0);  mul_tensor_7 = None
        unsqueeze_default_8: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        mul_tensor_8: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00015943877551020407);  sum_dim_int_list_1 = None
        squeeze_dims_1: "f32[1280]" = torch.ops.aten.squeeze.dims(arg407_1, [0, 2, 3]);  arg407_1 = None
        mul_tensor_9: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_10: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        unsqueeze_default_10: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, 0);  mul_tensor_10 = None
        unsqueeze_default_11: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        mul_tensor_11: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg161_1);  squeeze_dims_1 = arg161_1 = None
        unsqueeze_default_13: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_tensor_11, 0);  mul_tensor_11 = None
        unsqueeze_default_14: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        mul_tensor_12: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_12);  sub_tensor_2 = unsqueeze_default_12 = None
        sub_tensor_3: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(mul_tensor_5, mul_tensor_12);  mul_tensor_5 = mul_tensor_12 = None
        sub_tensor_4: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_tensor_3, unsqueeze_default_9);  sub_tensor_3 = unsqueeze_default_9 = None
        mul_tensor_13: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_4, unsqueeze_default_15);  sub_tensor_4 = unsqueeze_default_15 = None
        return mul_tensor_13



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
