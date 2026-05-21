"""
Standalone repro captured via capture_hook.
Label: torchbench_timm_resnest_train_000
Pattern hash: c177cfce7234
Shape hash: fd4f7fd1
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
_shapes_config = "(T([32, 1024, 14, 14], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), S([32, 2, 512, 14, 14]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_22: "f32[32, 1024, 14, 14]", arg126_1: "f32[1024]", arg127_1: "f32[1024]", arg128_1: "f32[1024]", arg129_1: "f32[1024]", _shape_param_0):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 1024, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 1024, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_1);  convolution_22 = None
        mul_tensor: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[1024]" = torch.ops.aten.mul.Tensor(arg126_1, 0.9)
        add_tensor_1: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0001594642002871);  squeeze_dims_1 = None
        mul_tensor_4: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[1024]" = torch.ops.aten.mul.Tensor(arg127_1, 0.9)
        add_tensor_2: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_default_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        view_default: "f32[32, 2, 512, 14, 14]" = torch.ops.aten.view.default(relu_default, _shape_param_0);  relu_default = _shape_param_0 = None
        sum_dim_int_list: "f32[32, 512, 14, 14]" = torch.ops.aten.sum.dim_IntList(view_default, [1]);  view_default = None
        mean_dim: "f32[32, 512, 1, 1]" = torch.ops.aten.mean.dim(sum_dim_int_list, [2, 3], True);  sum_dim_int_list = None
        copy__default: "f32[1024]" = torch.ops.aten.copy_.default(arg126_1, add_tensor_1);  arg126_1 = add_tensor_1 = None
        copy__default_1: "f32[1024]" = torch.ops.aten.copy_.default(arg127_1, add_tensor_2);  arg127_1 = add_tensor_2 = None
        return (mean_dim, copy__default, copy__default_1)



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
