"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train_000
Pattern hash: bff30f530d6e
Shape hash: 517912ff
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
_shapes_config = "(T([64, 512, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([640], f32), T([640], f32), T([640], f32), T([640], f32))"

class Repro(torch.nn.Module):
    def forward(self, avg_pool2d_2: "f32[64, 512, 7, 7]", convolution_89: "f32[64, 32, 7, 7]", convolution_91: "f32[64, 32, 7, 7]", convolution_93: "f32[64, 32, 7, 7]", convolution_95: "f32[64, 32, 7, 7]", arg578_1: "f32[640]", arg579_1: "f32[640]", arg580_1: "f32[640]", arg581_1: "f32[640]"):
        # No stacktrace found for following nodes
        cat_default: "f32[64, 640, 7, 7]" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95], 1);  avg_pool2d_2 = convolution_89 = convolution_91 = convolution_93 = convolution_95 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(cat_default, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 640, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 640, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 640, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 640, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[64, 640, 7, 7]" = torch.ops.aten.sub.Tensor(cat_default, getitem_1);  cat_default = None
        mul_tensor: "f32[64, 640, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_1: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_2: "f32[640]" = torch.ops.aten.mul.Tensor(arg578_1, 0.9)
        add_tensor_1: "f32[640]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_2: "f32[640]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0003189792663476);  squeeze_dims_2 = None
        mul_tensor_4: "f32[640]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[640]" = torch.ops.aten.mul.Tensor(arg579_1, 0.9)
        add_tensor_2: "f32[640]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(arg580_1, -1);  arg580_1 = None
        unsqueeze_default_1: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[64, 640, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(arg581_1, -1);  arg581_1 = None
        unsqueeze_default_3: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[64, 640, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[64, 640, 7, 7]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        unsqueeze_default_4: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        copy__default: "f32[640]" = torch.ops.aten.copy_.default(arg578_1, add_tensor_1);  arg578_1 = add_tensor_1 = None
        copy__default_1: "f32[640]" = torch.ops.aten.copy_.default(arg579_1, add_tensor_2);  arg579_1 = add_tensor_2 = None
        return (squeeze_dims_1, relu_default, unsqueeze_default_6, copy__default, copy__default_1)



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
