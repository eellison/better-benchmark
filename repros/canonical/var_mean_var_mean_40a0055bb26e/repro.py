"""
Standalone repro captured via capture_hook.
Label: torchbench_shufflenet_v2_x1_0_train_000
Pattern hash: 40a0055bb26e
Shape hash: 4e7adfa3
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([512, 232, 7, 7], f32), T([232], f32), T([232], f32), T([232], f32), T([232], f32), T([512, 232, 7, 7], f32), T([232], f32), T([232], f32), T([232], f32), T([232], f32), S([512, 2, 232, 7, 7]), S([512, 464, 7, 7]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_42: "f32[512, 232, 7, 7]", arg255_1: "f32[232]", arg256_1: "f32[232]", arg257_1: "f32[232]", arg258_1: "f32[232]", convolution_45: "f32[512, 232, 7, 7]", arg273_1: "f32[232]", arg274_1: "f32[232]", arg275_1: "f32[232]", arg276_1: "f32[232]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 232, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 232, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 232, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 232, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_1);  convolution_42 = None
        mul_tensor: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[232]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[232]" = torch.ops.aten.mul.Tensor(arg255_1, 0.9)
        add_tensor_1: "f32[232]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[232]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0000398612827361);  squeeze_dims_1 = None
        mul_tensor_4: "f32[232]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[232]" = torch.ops.aten.mul.Tensor(arg256_1, 0.9)
        add_tensor_2: "f32[232]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_default_1: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg258_1, -1);  arg258_1 = None
        unsqueeze_default_3: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[512, 232, 7, 7]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 232, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 232, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4: "f32[1, 232, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_default_1: "f32[1, 232, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_3);  convolution_45 = None
        mul_tensor_7: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        squeeze_dims_2: "f32[232]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        mul_tensor_8: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 0.1);  squeeze_dims_2 = None
        mul_tensor_9: "f32[232]" = torch.ops.aten.mul.Tensor(arg273_1, 0.9)
        add_tensor_5: "f32[232]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        squeeze_dims_3: "f32[232]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_10: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 1.0000398612827361);  squeeze_dims_3 = None
        mul_tensor_11: "f32[232]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.1);  mul_tensor_10 = None
        mul_tensor_12: "f32[232]" = torch.ops.aten.mul.Tensor(arg274_1, 0.9)
        add_tensor_6: "f32[232]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_4: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_default_5: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_13: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5);  mul_tensor_7 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_default_7: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_7: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7);  mul_tensor_13 = unsqueeze_default_7 = None
        relu_default_1: "f32[512, 232, 7, 7]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None
        cat_default: "f32[512, 464, 7, 7]" = torch.ops.aten.cat.default([relu_default, relu_default_1], 1);  relu_default = relu_default_1 = None
        view_default: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.view.default(cat_default, _shape_param_0);  cat_default = _shape_param_0 = None
        permute_default: "f32[512, 232, 2, 7, 7]" = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3, 4]);  view_default = None
        clone_default: "f32[512, 232, 2, 7, 7]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1: "f32[512, 464, 7, 7]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        split_tensor = torch.ops.aten.split.Tensor(view_default_1, 232, 1);  view_default_1 = None
        getitem_4: "f32[512, 232, 7, 7]" = split_tensor[0]
        getitem_5: "f32[512, 232, 7, 7]" = split_tensor[1];  split_tensor = None
        copy__default: "f32[232]" = torch.ops.aten.copy_.default(arg255_1, add_tensor_1);  arg255_1 = add_tensor_1 = None
        copy__default_1: "f32[232]" = torch.ops.aten.copy_.default(arg256_1, add_tensor_2);  arg256_1 = add_tensor_2 = None
        copy__default_2: "f32[232]" = torch.ops.aten.copy_.default(arg273_1, add_tensor_5);  arg273_1 = add_tensor_5 = None
        copy__default_3: "f32[232]" = torch.ops.aten.copy_.default(arg274_1, add_tensor_6);  arg274_1 = add_tensor_6 = None
        return (getitem_5, copy__default, copy__default_1, copy__default_2, copy__default_3, getitem_4)

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
