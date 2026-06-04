"""
Standalone repro captured via capture_hook.
Label: timm_timm_visformer_small_train_train_000
Pattern hash: aabd7680d8e0
Shape hash: 540a56ff
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 768, 7, 7], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32), T([1, 768, 7, 7], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_40: "f32[128, 768, 7, 7]", arg138_1: "f32[768]", arg139_1: "f32[768]", arg140_1: "f32[768]", arg141_1: "f32[768]", arg142_1: "f32[1, 768, 7, 7]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[768]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 768, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 768, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_1);  convolution_40 = None
        mul_tensor: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[768]" = torch.ops.aten.mul.Tensor(arg138_1, 0.9)
        add_tensor_1: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[768]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0001594642002871);  squeeze_dims_1 = None
        mul_tensor_4: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[768]" = torch.ops.aten.mul.Tensor(arg139_1, 0.9)
        add_tensor_2: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_default_1: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_default_3: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        add_tensor_4: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_3, arg142_1);  add_tensor_3 = arg142_1 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 768, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 768, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_5: "f32[1, 768, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_default_1: "f32[1, 768, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        sub_tensor_1: "f32[128, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_tensor_4, getitem_3);  add_tensor_4 = None
        mul_tensor_7: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        squeeze_dims_2: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_dims_3: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_default_1, [0, 2, 3]);  rsqrt_default_1 = None
        mul_tensor_8: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 0.1)
        mul_tensor_9: "f32[768]" = torch.ops.aten.mul.Tensor(arg144_1, 0.9)
        add_tensor_6: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        squeeze_dims_4: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_10: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_4, 1.0001594642002871);  squeeze_dims_4 = None
        mul_tensor_11: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.1);  mul_tensor_10 = None
        mul_tensor_12: "f32[768]" = torch.ops.aten.mul.Tensor(arg145_1, 0.9)
        add_tensor_7: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_4: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_default_5: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_13: "f32[128, 768, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5);  mul_tensor_7 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_default_7: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_8: "f32[128, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7);  mul_tensor_13 = unsqueeze_default_7 = None
        unsqueeze_default_8: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_9: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 3);  unsqueeze_default_9 = None
        copy__default: "f32[768]" = torch.ops.aten.copy_.default(arg138_1, add_tensor_1);  arg138_1 = add_tensor_1 = None
        copy__default_1: "f32[768]" = torch.ops.aten.copy_.default(arg139_1, add_tensor_2);  arg139_1 = add_tensor_2 = None
        copy__default_2: "f32[768]" = torch.ops.aten.copy_.default(arg144_1, add_tensor_6);  arg144_1 = add_tensor_6 = None
        copy__default_3: "f32[768]" = torch.ops.aten.copy_.default(arg145_1, add_tensor_7);  arg145_1 = add_tensor_7 = None
        return (squeeze_dims_3, add_tensor_8, unsqueeze_default_10, copy__default, copy__default_1, copy__default_2, copy__default_3)

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
