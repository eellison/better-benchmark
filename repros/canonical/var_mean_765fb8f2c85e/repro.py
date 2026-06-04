"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train_000
Pattern hash: 765fb8f2c85e
Shape hash: dae27bc8
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 192, 71, 71], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_4: "f32[128, 192, 71, 71]", arg27_1: "f32[192]", arg28_1: "f32[192]", arg29_1: "f32[192]", arg30_1: "f32[192]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 192, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 192, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt_default: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 192, 71, 71]" = torch.ops.aten.sub.Tensor(convolution_4, getitem_1);  convolution_4 = None
        mul_tensor: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[192]" = torch.ops.aten.mul.Tensor(arg27_1, 0.9)
        add_tensor_1: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[192]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.00000154979411);  squeeze_dims_1 = None
        mul_tensor_4: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[192]" = torch.ops.aten.mul.Tensor(arg28_1, 0.9)
        add_tensor_2: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 192, 71, 71]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 192, 71, 71]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[128, 192, 71, 71]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_default = None
        getitem_2: "f32[128, 192, 35, 35]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_3: "i8[128, 192, 35, 35]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        avg_pool2d_default: "f32[128, 192, 35, 35]" = torch.ops.aten.avg_pool2d.default(getitem_2, [3, 3], [1, 1], [1, 1]);  getitem_2 = None
        copy__default: "f32[192]" = torch.ops.aten.copy_.default(arg27_1, add_tensor_1);  arg27_1 = add_tensor_1 = None
        copy__default_1: "f32[192]" = torch.ops.aten.copy_.default(arg28_1, add_tensor_2);  arg28_1 = add_tensor_2 = None
        return (getitem_3, avg_pool2d_default, copy__default, copy__default_1)

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
