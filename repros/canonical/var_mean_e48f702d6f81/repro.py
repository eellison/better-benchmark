"""
Standalone repro captured via capture_hook.
Label: torchbench_functorch_dp_cifar10_train_000
Pattern hash: e48f702d6f81
Shape hash: 90159c73
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 512, 1, 1], f32), T([512], f32), T([512], f32), S([64, 32, 16, 1]), S([64, 512, 1, 1]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_18: "f32[64, 512, 1, 1]", arg56_1: "f32[512]", arg57_1: "f32[512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[64, 32, 16, 1]" = torch.ops.aten.view.default(convolution_18, _shape_param_0);  convolution_18 = _shape_param_0 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(view_default, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[64, 32, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 32, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[64, 32, 16, 1]" = torch.ops.aten.sub.Tensor(view_default, getitem_1);  view_default = None
        mul_tensor: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        view_default_1: "f32[64, 512, 1, 1]" = torch.ops.aten.view.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None
        unsqueeze_default: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(arg56_1, 0);  arg56_1 = None
        unsqueeze_default_1: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_1: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(view_default_1, unsqueeze_default_2);  view_default_1 = unsqueeze_default_2 = None
        unsqueeze_default_3: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(arg57_1, 0);  arg57_1 = None
        unsqueeze_default_4: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        add_tensor_1: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        squeeze_dims: "f32[64, 32]" = torch.ops.aten.squeeze.dims(getitem_1, [2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[64, 32]" = torch.ops.aten.squeeze.dims(rsqrt_default, [2, 3]);  rsqrt_default = None
        relu_default: "f32[64, 512, 1, 1]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        return (squeeze_dims, squeeze_dims_1, relu_default)

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
