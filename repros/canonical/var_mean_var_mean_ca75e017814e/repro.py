"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train_000
Pattern hash: ca75e017814e
Shape hash: 971fedf8
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 320, 8, 8], f32), T([320], f32), T([320], f32), T([320], f32), T([320], f32), T([128, 192, 8, 8], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), T([128, 768, 17, 17], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_71: "f32[128, 320, 8, 8]", arg429_1: "f32[320]", arg430_1: "f32[320]", arg431_1: "f32[320]", arg432_1: "f32[320]", convolution_75: "f32[128, 192, 8, 8]", arg453_1: "f32[192]", arg454_1: "f32[192]", arg455_1: "f32[192]", arg456_1: "f32[192]", cat_7: "f32[128, 768, 17, 17]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 320, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 320, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt_default: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_1);  convolution_71 = None
        mul_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[320]" = torch.ops.aten.mul.Tensor(arg429_1, 0.9)
        add_tensor_1: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[320]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0001220852154804);  squeeze_dims_1 = None
        mul_tensor_4: "f32[320]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[320]" = torch.ops.aten.mul.Tensor(arg430_1, 0.9)
        add_tensor_2: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg431_1, -1);  arg431_1 = None
        unsqueeze_default_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(arg432_1, -1);  arg432_1 = None
        unsqueeze_default_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[128, 320, 8, 8]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 192, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 192, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
        rsqrt_default_1: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_3);  convolution_75 = None
        mul_tensor_7: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        squeeze_dims_2: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        mul_tensor_8: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 0.1);  squeeze_dims_2 = None
        mul_tensor_9: "f32[192]" = torch.ops.aten.mul.Tensor(arg453_1, 0.9)
        add_tensor_5: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        squeeze_dims_3: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_10: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 1.0001220852154804);  squeeze_dims_3 = None
        mul_tensor_11: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.1);  mul_tensor_10 = None
        mul_tensor_12: "f32[192]" = torch.ops.aten.mul.Tensor(arg454_1, 0.9)
        add_tensor_6: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_4: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg455_1, -1);  arg455_1 = None
        unsqueeze_default_5: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_13: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5);  mul_tensor_7 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg456_1, -1);  arg456_1 = None
        unsqueeze_default_7: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_7: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7);  mul_tensor_13 = unsqueeze_default_7 = None
        relu_default_1: "f32[128, 192, 8, 8]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False);  cat_7 = None
        getitem_4: "f32[128, 768, 8, 8]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_5: "i8[128, 768, 8, 8]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        cat_default: "f32[128, 1280, 8, 8]" = torch.ops.aten.cat.default([relu_default, relu_default_1, getitem_4], 1);  relu_default = relu_default_1 = getitem_4 = None
        avg_pool2d_default: "f32[128, 1280, 8, 8]" = torch.ops.aten.avg_pool2d.default(cat_default, [3, 3], [1, 1], [1, 1]);  cat_default = None
        copy__default: "f32[320]" = torch.ops.aten.copy_.default(arg429_1, add_tensor_1);  arg429_1 = add_tensor_1 = None
        copy__default_1: "f32[320]" = torch.ops.aten.copy_.default(arg430_1, add_tensor_2);  arg430_1 = add_tensor_2 = None
        copy__default_2: "f32[192]" = torch.ops.aten.copy_.default(arg453_1, add_tensor_5);  arg453_1 = add_tensor_5 = None
        copy__default_3: "f32[192]" = torch.ops.aten.copy_.default(arg454_1, add_tensor_6);  arg454_1 = add_tensor_6 = None
        return (getitem_5, avg_pool2d_default, copy__default, copy__default_1, copy__default_2, copy__default_3)

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
