"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train_000
Pattern hash: 0407b3e7c77f
Shape hash: 225bfb21
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 192, 17, 17], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([128, 192, 17, 17], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([128, 192, 17, 17], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([128, 192, 17, 17], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_50: "f32[128, 192, 17, 17]", arg303_1: "f32[192]", arg304_1: "f32[192]", arg305_1: "f32[192]", arg306_1: "f32[192]", convolution_53: "f32[128, 192, 17, 17]", arg321_1: "f32[192]", arg322_1: "f32[192]", arg323_1: "f32[192]", arg324_1: "f32[192]", convolution_58: "f32[128, 192, 17, 17]", arg351_1: "f32[192]", arg352_1: "f32[192]", arg353_1: "f32[192]", arg354_1: "f32[192]", convolution_59: "f32[128, 192, 17, 17]", arg357_1: "f32[192]", arg358_1: "f32[192]", arg359_1: "f32[192]", arg360_1: "f32[192]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 192, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 192, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt_default: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_1);  convolution_50 = None
        mul_tensor: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[192]" = torch.ops.aten.mul.Tensor(arg303_1, 0.9)
        add_tensor_1: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[192]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0000270336027683);  squeeze_dims_1 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(arg304_1, 0.9)
        add_tensor_2: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg306_1, -1);  arg306_1 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 192, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 192, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
        rsqrt_default_1: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_3);  convolution_53 = None
        mul_tensor_7: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        squeeze_dims_2: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        mul_tensor_8: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 0.1);  squeeze_dims_2 = None
        mul_tensor_9: "f32[192]" = torch.ops.aten.mul.Tensor(arg321_1, 0.9)
        add_tensor_5: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        squeeze_dims_3: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_10: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 1.0000270336027683);  squeeze_dims_3 = None
        mul_tensor_11: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.1);  mul_tensor_10 = None
        mul_tensor_12: "f32[192]" = torch.ops.aten.mul.Tensor(arg322_1, 0.9)
        add_tensor_6: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_4: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
        unsqueeze_default_5: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_13: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5);  mul_tensor_7 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_default_7: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_7: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7);  mul_tensor_13 = unsqueeze_default_7 = None
        relu_default_1: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 192, 1, 1]" = var_mean_correction_2[0]
        getitem_5: "f32[1, 192, 1, 1]" = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_8: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
        rsqrt_default_2: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        sub_tensor_2: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_5);  convolution_58 = None
        mul_tensor_14: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = rsqrt_default_2 = None
        squeeze_dims_4: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        mul_tensor_15: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_4, 0.1);  squeeze_dims_4 = None
        mul_tensor_16: "f32[192]" = torch.ops.aten.mul.Tensor(arg351_1, 0.9)
        add_tensor_9: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_5: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_17: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.0000270336027683);  squeeze_dims_5 = None
        mul_tensor_18: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[192]" = torch.ops.aten.mul.Tensor(arg352_1, 0.9)
        add_tensor_10: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
        unsqueeze_default_8: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg353_1, -1);  arg353_1 = None
        unsqueeze_default_9: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        mul_tensor_20: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_14, unsqueeze_default_9);  mul_tensor_14 = unsqueeze_default_9 = None
        unsqueeze_default_10: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_default_11: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        add_tensor_11: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_20, unsqueeze_default_11);  mul_tensor_20 = unsqueeze_default_11 = None
        relu_default_2: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_11);  add_tensor_11 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 192, 1, 1]" = var_mean_correction_3[0]
        getitem_7: "f32[1, 192, 1, 1]" = var_mean_correction_3[1];  var_mean_correction_3 = None
        add_tensor_12: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 0.001)
        rsqrt_default_3: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_12);  add_tensor_12 = None
        sub_tensor_3: "f32[128, 192, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_7);  convolution_59 = None
        mul_tensor_21: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_3);  sub_tensor_3 = rsqrt_default_3 = None
        squeeze_dims_6: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        mul_tensor_22: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 0.1);  squeeze_dims_6 = None
        mul_tensor_23: "f32[192]" = torch.ops.aten.mul.Tensor(arg357_1, 0.9)
        add_tensor_13: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_22, mul_tensor_23);  mul_tensor_22 = mul_tensor_23 = None
        squeeze_dims_7: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_24: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, 1.0000270336027683);  squeeze_dims_7 = None
        mul_tensor_25: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_24, 0.1);  mul_tensor_24 = None
        mul_tensor_26: "f32[192]" = torch.ops.aten.mul.Tensor(arg358_1, 0.9)
        add_tensor_14: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        unsqueeze_default_12: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_default_13: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_27: "f32[128, 192, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_21, unsqueeze_default_13);  mul_tensor_21 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_default_15: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_15: "f32[128, 192, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_27, unsqueeze_default_15);  mul_tensor_27 = unsqueeze_default_15 = None
        relu_default_3: "f32[128, 192, 17, 17]" = torch.ops.aten.relu.default(add_tensor_15);  add_tensor_15 = None
        cat_default: "f32[128, 768, 17, 17]" = torch.ops.aten.cat.default([relu_default, relu_default_1, relu_default_2, relu_default_3], 1);  relu_default = relu_default_1 = relu_default_2 = relu_default_3 = None
        avg_pool2d_default: "f32[128, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_default, [3, 3], [1, 1], [1, 1]);  cat_default = None
        copy__default: "f32[192]" = torch.ops.aten.copy_.default(arg303_1, add_tensor_1);  arg303_1 = add_tensor_1 = None
        copy__default_1: "f32[192]" = torch.ops.aten.copy_.default(arg304_1, add_tensor_2);  arg304_1 = add_tensor_2 = None
        copy__default_2: "f32[192]" = torch.ops.aten.copy_.default(arg321_1, add_tensor_5);  arg321_1 = add_tensor_5 = None
        copy__default_3: "f32[192]" = torch.ops.aten.copy_.default(arg322_1, add_tensor_6);  arg322_1 = add_tensor_6 = None
        copy__default_4: "f32[192]" = torch.ops.aten.copy_.default(arg351_1, add_tensor_9);  arg351_1 = add_tensor_9 = None
        copy__default_5: "f32[192]" = torch.ops.aten.copy_.default(arg352_1, add_tensor_10);  arg352_1 = add_tensor_10 = None
        copy__default_6: "f32[192]" = torch.ops.aten.copy_.default(arg357_1, add_tensor_13);  arg357_1 = add_tensor_13 = None
        copy__default_7: "f32[192]" = torch.ops.aten.copy_.default(arg358_1, add_tensor_14);  arg358_1 = add_tensor_14 = None
        return (avg_pool2d_default, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7)

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
