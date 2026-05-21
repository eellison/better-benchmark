"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train_000
Pattern hash: 65e90900fd65
Shape hash: 50a24a6c
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
_shapes_config = "(T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_90: "f32[128, 384, 8, 8]", arg543_1: "f32[384]", arg544_1: "f32[384]", arg545_1: "f32[384]", arg546_1: "f32[384]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_90, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 384, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt_default: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_1);  convolution_90 = None
        mul_tensor: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_1: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_2: "f32[384]" = torch.ops.aten.mul.Tensor(arg543_1, 0.9)
        add_tensor_1: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_2: "f32[384]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0001220852154804);  squeeze_dims_2 = None
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(arg544_1, 0.9)
        add_tensor_2: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg545_1, -1);  arg545_1 = None
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg546_1, -1);  arg546_1 = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        unsqueeze_default_4: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        copy__default: "f32[384]" = torch.ops.aten.copy_.default(arg543_1, add_tensor_1);  arg543_1 = add_tensor_1 = None
        copy__default_1: "f32[384]" = torch.ops.aten.copy_.default(arg544_1, add_tensor_2);  arg544_1 = add_tensor_2 = None
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
