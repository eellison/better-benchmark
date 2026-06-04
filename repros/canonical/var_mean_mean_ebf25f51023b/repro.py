"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_train_000
Pattern hash: ebf25f51023b
Shape hash: 312b67db
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([256, 960, 7, 7], f32), T([960], f32), T([960], f32), T([960], f32), T([960], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_57: "f32[256, 960, 7, 7]", arg289_1: "f32[960]", arg290_1: "f32[960]", arg291_1: "f32[960]", arg292_1: "f32[960]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 960, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 960, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 960, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt_default: "f32[1, 960, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_1);  convolution_57 = None
        mul_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.01);  squeeze_dims = None
        mul_tensor_2: "f32[960]" = torch.ops.aten.mul.Tensor(arg289_1, 0.99)
        add_tensor_1: "f32[960]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[960]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0000797257434426);  squeeze_dims_1 = None
        mul_tensor_4: "f32[960]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.01);  mul_tensor_3 = None
        mul_tensor_5: "f32[960]" = torch.ops.aten.mul.Tensor(arg290_1, 0.99)
        add_tensor_2: "f32[960]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg291_1, -1);  arg291_1 = None
        unsqueeze_default_1: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[960, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_default_3: "f32[960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[256, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        add_tensor_4: "f32[256, 960, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_3, 3)
        clamp_min_default: "f32[256, 960, 7, 7]" = torch.ops.aten.clamp_min.default(add_tensor_4, 0);  add_tensor_4 = None
        clamp_max_default: "f32[256, 960, 7, 7]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor_7: "f32[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_tensor_3, clamp_max_default);  add_tensor_3 = clamp_max_default = None
        div_tensor: "f32[256, 960, 7, 7]" = torch.ops.aten.div.Tensor(mul_tensor_7, 6);  mul_tensor_7 = None
        mean_dim: "f32[256, 960, 1, 1]" = torch.ops.aten.mean.dim(div_tensor, [-1, -2], True);  div_tensor = None
        copy__default: "f32[960]" = torch.ops.aten.copy_.default(arg289_1, add_tensor_1);  arg289_1 = add_tensor_1 = None
        copy__default_1: "f32[960]" = torch.ops.aten.copy_.default(arg290_1, add_tensor_2);  arg290_1 = add_tensor_2 = None
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
