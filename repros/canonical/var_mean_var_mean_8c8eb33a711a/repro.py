"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_train_000
Pattern hash: 8c8eb33a711a
Shape hash: e37b62dc
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([32, 1024, 14, 14], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([32, 1024, 14, 14], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_19: "f32[32, 1024, 14, 14]", arg108_1: "f32[1024]", arg109_1: "f32[1024]", arg110_1: "f32[1024]", arg111_1: "f32[1024]", convolution_20: "f32[32, 1024, 14, 14]", arg114_1: "f32[1024]", arg115_1: "f32[1024]", arg116_1: "f32[1024]", arg117_1: "f32[1024]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 1024, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1024, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_1);  convolution_19 = None
        mul_tensor: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_1: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_2: "f32[1024]" = torch.ops.aten.mul.Tensor(arg108_1, 0.9)
        add_tensor_1: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_2: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0001594642002871);  squeeze_dims_2 = None
        mul_tensor_4: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[1024]" = torch.ops.aten.mul.Tensor(arg109_1, 0.9)
        add_tensor_2: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_default_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 1024, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 1024, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_default_1: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_3);  convolution_20 = None
        mul_tensor_7: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        squeeze_dims_3: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_dims_4: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_default_1, [0, 2, 3]);  rsqrt_default_1 = None
        mul_tensor_8: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.1)
        mul_tensor_9: "f32[1024]" = torch.ops.aten.mul.Tensor(arg114_1, 0.9)
        add_tensor_5: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        squeeze_dims_5: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_10: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.0001594642002871);  squeeze_dims_5 = None
        mul_tensor_11: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.1);  mul_tensor_10 = None
        mul_tensor_12: "f32[1024]" = torch.ops.aten.mul.Tensor(arg115_1, 0.9)
        add_tensor_6: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_4: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_default_5: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_13: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5);  mul_tensor_7 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_default_7: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_7: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7);  mul_tensor_13 = unsqueeze_default_7 = None
        add_tensor_8: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_3, add_tensor_7);  add_tensor_3 = add_tensor_7 = None
        relu_default: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_tensor_8);  add_tensor_8 = None
        avg_pool2d_default: "f32[32, 1024, 7, 7]" = torch.ops.aten.avg_pool2d.default(relu_default, [2, 2], [2, 2], [0, 0], True, False);  relu_default = None
        unsqueeze_default_8: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_dims_3, 0);  squeeze_dims_3 = None
        unsqueeze_default_9: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 3);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_12: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 2);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 3);  unsqueeze_default_12 = None
        copy__default: "f32[1024]" = torch.ops.aten.copy_.default(arg108_1, add_tensor_1);  arg108_1 = add_tensor_1 = None
        copy__default_1: "f32[1024]" = torch.ops.aten.copy_.default(arg109_1, add_tensor_2);  arg109_1 = add_tensor_2 = None
        copy__default_2: "f32[1024]" = torch.ops.aten.copy_.default(arg114_1, add_tensor_5);  arg114_1 = add_tensor_5 = None
        copy__default_3: "f32[1024]" = torch.ops.aten.copy_.default(arg115_1, add_tensor_6);  arg115_1 = add_tensor_6 = None
        return (squeeze_dims_1, squeeze_dims_4, avg_pool2d_default, unsqueeze_default_10, unsqueeze_default_13, copy__default, copy__default_1, copy__default_2, copy__default_3)

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
