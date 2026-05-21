"""
Standalone repro captured via capture_hook.
Label: timm_mobilenetv2_100_train_000
Pattern hash: bbc2387d0095
Shape hash: 74c967c3
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
_shapes_config = "(T([128, 160, 7, 7], f32), T([160], f32), T([160], f32), T([160], f32), T([160], f32), T([128, 160, 7, 7], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_47: "f32[128, 160, 7, 7]", arg285_1: "f32[160]", arg286_1: "f32[160]", arg287_1: "f32[160]", arg288_1: "f32[160]", add_233: "f32[128, 160, 7, 7]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_47, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 160, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 160, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 160, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 160, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, getitem_1);  convolution_47 = None
        mul_tensor: "f32[128, 160, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_1: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_2: "f32[160]" = torch.ops.aten.mul.Tensor(arg285_1, 0.9)
        add_tensor_1: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_2: "f32[160]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0001594642002871);  squeeze_dims_2 = None
        mul_tensor_4: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[160]" = torch.ops.aten.mul.Tensor(arg286_1, 0.9)
        add_tensor_2: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_default_1: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 160, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_default_3: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 160, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        add_tensor_4: "f32[128, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_3, add_233);  add_tensor_3 = add_233 = None
        unsqueeze_default_4: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_5: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        copy__default: "f32[160]" = torch.ops.aten.copy_.default(arg285_1, add_tensor_1);  arg285_1 = add_tensor_1 = None
        copy__default_1: "f32[160]" = torch.ops.aten.copy_.default(arg286_1, add_tensor_2);  arg286_1 = add_tensor_2 = None
        return (squeeze_dims_1, add_tensor_4, unsqueeze_default_6, copy__default, copy__default_1)



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
