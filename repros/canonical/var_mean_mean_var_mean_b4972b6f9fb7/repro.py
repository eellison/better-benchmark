"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train_000
Pattern hash: b4972b6f9fb7
Shape hash: 87b46cfc
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
_shapes_config = "(T([128, 1408, 7, 7], f32), T([1408], f32), T([1408], f32), T([1408], f32), T([1408], f32), T([128, 1408, 7, 7], f32), T([1408], f32), T([1408], f32), T([1408], f32), T([1408], f32), S([128, 1408]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_42: "f32[128, 1408, 7, 7]", arg340_1: "f32[1408]", arg341_1: "f32[1408]", arg342_1: "f32[1408]", arg343_1: "f32[1408]", convolution_43: "f32[128, 1408, 7, 7]", arg346_1: "f32[1408]", arg347_1: "f32[1408]", arg348_1: "f32[1408]", arg349_1: "f32[1408]", _shape_param_0):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 1408, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1408, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 1408, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 1408, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_1);  convolution_42 = None
        mul_tensor: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[1408]" = torch.ops.aten.mul.Tensor(arg340_1, 0.9)
        add_tensor_1: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0001594642002871);  squeeze_dims_1 = None
        mul_tensor_4: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[1408]" = torch.ops.aten.mul.Tensor(arg341_1, 0.9)
        add_tensor_2: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg342_1, -1);  arg342_1 = None
        unsqueeze_default_1: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg343_1, -1);  arg343_1 = None
        unsqueeze_default_3: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 1408, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 1408, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4: "f32[1, 1408, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_default_1: "f32[1, 1408, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1: "f32[128, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_3);  convolution_43 = None
        mul_tensor_7: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        squeeze_dims_2: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        mul_tensor_8: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 0.1);  squeeze_dims_2 = None
        mul_tensor_9: "f32[1408]" = torch.ops.aten.mul.Tensor(arg346_1, 0.9)
        add_tensor_5: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        squeeze_dims_3: "f32[1408]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_10: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 1.0001594642002871);  squeeze_dims_3 = None
        mul_tensor_11: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.1);  mul_tensor_10 = None
        mul_tensor_12: "f32[1408]" = torch.ops.aten.mul.Tensor(arg347_1, 0.9)
        add_tensor_6: "f32[1408]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_4: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg348_1, -1);  arg348_1 = None
        unsqueeze_default_5: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_13: "f32[128, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5);  mul_tensor_7 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1408, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_default_7: "f32[1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_7: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7);  mul_tensor_13 = unsqueeze_default_7 = None
        add_tensor_8: "f32[128, 1408, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_3, add_tensor_7);  add_tensor_3 = add_tensor_7 = None
        relu_default: "f32[128, 1408, 7, 7]" = torch.ops.aten.relu.default(add_tensor_8);  add_tensor_8 = None
        mean_dim: "f32[128, 1408, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None
        as_strided_default: "f32[128, 1408, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 1408, 1, 1], [1408, 1, 1408, 1408]);  mean_dim = None
        view_default: "f32[128, 1408]" = torch.ops.aten.view.default(as_strided_default, _shape_param_0);  as_strided_default = _shape_param_0 = None
        copy__default: "f32[1408]" = torch.ops.aten.copy_.default(arg340_1, add_tensor_1);  arg340_1 = add_tensor_1 = None
        copy__default_1: "f32[1408]" = torch.ops.aten.copy_.default(arg341_1, add_tensor_2);  arg341_1 = add_tensor_2 = None
        copy__default_2: "f32[1408]" = torch.ops.aten.copy_.default(arg346_1, add_tensor_5);  arg346_1 = add_tensor_5 = None
        copy__default_3: "f32[1408]" = torch.ops.aten.copy_.default(arg347_1, add_tensor_6);  arg347_1 = add_tensor_6 = None
        return (copy__default, copy__default_1, view_default, copy__default_2, copy__default_3)



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
