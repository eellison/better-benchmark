"""
Standalone repro captured via capture_hook.
Label: torchbench_LearningToPaint_train_000
Pattern hash: 5b0c15caa439
Shape hash: cd3b9d56
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
_shapes_config = "(T([1024, 512, 4, 4], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([1024, 512, 4, 4], f32), S([1024, -1]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_20: "f32[1024, 512, 4, 4]", arg123_1: "f32[512]", arg124_1: "f32[512]", arg125_1: "f32[512]", arg126_1: "f32[512]", relu_14: "f32[1024, 512, 4, 4]", _shape_param_0):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_1);  convolution_20 = None
        mul_tensor: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_1: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_2: "f32[512]" = torch.ops.aten.mul.Tensor(arg123_1, 0.9)
        add_tensor_1: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_2: "f32[512]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0000610388817677);  squeeze_dims_2 = None
        mul_tensor_4: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[512]" = torch.ops.aten.mul.Tensor(arg124_1, 0.9)
        add_tensor_2: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_default_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_default_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[1024, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        add_tensor_4: "f32[1024, 512, 4, 4]" = torch.ops.aten.add.Tensor(add_tensor_3, relu_14);  add_tensor_3 = relu_14 = None
        relu_default: "f32[1024, 512, 4, 4]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        avg_pool2d_default: "f32[1024, 512, 1, 1]" = torch.ops.aten.avg_pool2d.default(relu_default, [4, 4]);  relu_default = None
        view_default: "f32[1024, 512]" = torch.ops.aten.view.default(avg_pool2d_default, _shape_param_0);  avg_pool2d_default = _shape_param_0 = None
        unsqueeze_default_4: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        copy__default: "f32[512]" = torch.ops.aten.copy_.default(arg123_1, add_tensor_1);  arg123_1 = add_tensor_1 = None
        copy__default_1: "f32[512]" = torch.ops.aten.copy_.default(arg124_1, add_tensor_2);  arg124_1 = add_tensor_2 = None
        return (squeeze_dims_1, view_default, unsqueeze_default_6, copy__default, copy__default_1)



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
