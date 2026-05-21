"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train_001
Pattern hash: 7246c400da84
Shape hash: d2eb30d4
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
_shapes_config = "(T([128, 32, 112, 112], f32), T([128, 32, 1, 1], f32), T([128, 32, 1, 1], f32), T([128, 32, 112, 112], f32), T([128, 32, 112, 112], f32), T([1, 32, 1, 1], f32), T([128, 32, 112, 112], f32), T([1, 32, 1, 1], f32), T([32], f32), S([128, 32, 112, 112]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_228: "f32[128, 32, 112, 112]", sigmoid_15: "f32[128, 32, 1, 1]", getitem_234: "f32[128, 32, 1, 1]", add_148: "f32[128, 32, 112, 112]", add_147: "f32[128, 32, 112, 112]", arg170_1: "f32[1, 32, 1, 1]", arg169_1: "f32[128, 32, 112, 112]", arg171_1: "f32[1, 32, 1, 1]", arg4_1: "f32[32]", _shape_param_0):
        # No stacktrace found for following nodes
        mul_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_228, sigmoid_15);  getitem_228 = sigmoid_15 = None
        expand_default: "f32[128, 32, 112, 112]" = torch.ops.aten.expand.default(getitem_234, _shape_param_0);  getitem_234 = _shape_param_0 = None
        div_scalar: "f32[128, 32, 112, 112]" = torch.ops.aten.div.Scalar(expand_default, 12544);  expand_default = None
        add_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None
        reciprocal_default: "f32[128, 32, 112, 112]" = torch.ops.aten.reciprocal.default(add_148);  add_148 = None
        mul_tensor_1: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_2: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_1);  add_tensor = None
        sub_tensor: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(1, mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_3: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_147, sub_tensor);  add_147 = sub_tensor = None
        add_tensor_1: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_3, 1);  mul_tensor_3 = None
        mul_tensor_4: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1);  mul_tensor_2 = add_tensor_1 = None
        squeeze_dims: "f32[32]" = torch.ops.aten.squeeze.dims(arg170_1, [0, 2, 3]);  arg170_1 = None
        unsqueeze_default: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_1: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        sum_dim_int_list: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 2, 3])
        sub_tensor_1: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(arg169_1, unsqueeze_default_2);  arg169_1 = unsqueeze_default_2 = None
        mul_tensor_5: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor_4, sub_tensor_1)
        sum_dim_int_list_1: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 2, 3]);  mul_tensor_5 = None
        mul_tensor_6: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list, 6.228077168367346e-07);  sum_dim_int_list = None
        unsqueeze_default_3: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_4: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        mul_tensor_7: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 6.228077168367346e-07)
        squeeze_dims_1: "f32[32]" = torch.ops.aten.squeeze.dims(arg171_1, [0, 2, 3]);  arg171_1 = None
        mul_tensor_8: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, squeeze_dims_1)
        mul_tensor_9: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_8);  mul_tensor_7 = mul_tensor_8 = None
        unsqueeze_default_6: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_9, 0);  mul_tensor_9 = None
        unsqueeze_default_7: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None
        mul_tensor_10: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, arg4_1);  arg4_1 = None
        unsqueeze_default_9: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_10, 0);  mul_tensor_10 = None
        unsqueeze_default_10: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None
        mul_tensor_11: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_8);  sub_tensor_1 = unsqueeze_default_8 = None
        sub_tensor_2: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(mul_tensor_4, mul_tensor_11);  mul_tensor_4 = mul_tensor_11 = None
        sub_tensor_3: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_5);  sub_tensor_2 = unsqueeze_default_5 = None
        mul_tensor_12: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_11);  sub_tensor_3 = unsqueeze_default_11 = None
        mul_tensor_13: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, squeeze_dims_1);  sum_dim_int_list_1 = squeeze_dims_1 = None
        return (mul_tensor_12, mul_tensor_13)



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
