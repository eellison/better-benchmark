"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train_000
Pattern hash: de5c360021d3
Shape hash: 054dc4a4
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
_shapes_config = "(T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_14: "f32[128, 384, 14, 14]", arg107_1: "f32[384]", arg108_1: "f32[384]", arg109_1: "f32[384]", arg110_1: "f32[384]", convolution_15: "f32[128, 384, 14, 14]", arg113_1: "f32[384]", arg114_1: "f32[384]", arg115_1: "f32[384]", arg116_1: "f32[384]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 384, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_1);  convolution_14 = None
        mul_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_1: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_2: "f32[384]" = torch.ops.aten.mul.Tensor(arg107_1, 0.9)
        add_tensor_1: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_2: "f32[384]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0000398612827361);  squeeze_dims_2 = None
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(arg108_1, 0.9)
        add_tensor_2: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 384, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 384, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_default_1: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_3);  convolution_15 = None
        mul_tensor_7: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        squeeze_dims_3: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_dims_4: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_default_1, [0, 2, 3]);  rsqrt_default_1 = None
        mul_tensor_8: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.1)
        mul_tensor_9: "f32[384]" = torch.ops.aten.mul.Tensor(arg113_1, 0.9)
        add_tensor_5: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        squeeze_dims_5: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_10: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.0000398612827361);  squeeze_dims_5 = None
        mul_tensor_11: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.1);  mul_tensor_10 = None
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(arg114_1, 0.9)
        add_tensor_6: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_4: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_default_5: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_13: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5);  mul_tensor_7 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_default_7: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_7: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7);  mul_tensor_13 = unsqueeze_default_7 = None
        add_tensor_8: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_3, add_tensor_7);  add_tensor_3 = add_tensor_7 = None
        relu_default: "f32[128, 384, 14, 14]" = torch.ops.aten.relu.default(add_tensor_8);  add_tensor_8 = None
        unsqueeze_default_8: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_3, 0);  squeeze_dims_3 = None
        unsqueeze_default_9: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 3);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_12: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 2);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 3);  unsqueeze_default_12 = None
        copy__default: "f32[384]" = torch.ops.aten.copy_.default(arg107_1, add_tensor_1);  arg107_1 = add_tensor_1 = None
        copy__default_1: "f32[384]" = torch.ops.aten.copy_.default(arg108_1, add_tensor_2);  arg108_1 = add_tensor_2 = None
        copy__default_2: "f32[384]" = torch.ops.aten.copy_.default(arg113_1, add_tensor_5);  arg113_1 = add_tensor_5 = None
        copy__default_3: "f32[384]" = torch.ops.aten.copy_.default(arg114_1, add_tensor_6);  arg114_1 = add_tensor_6 = None
        return (squeeze_dims_1, squeeze_dims_4, relu_default, unsqueeze_default_10, unsqueeze_default_13, copy__default, copy__default_1, copy__default_2, copy__default_3)



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
