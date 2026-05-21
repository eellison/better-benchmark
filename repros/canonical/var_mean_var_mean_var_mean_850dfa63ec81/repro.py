"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train_000
Pattern hash: 850dfa63ec81
Shape hash: 9f8c2e21
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
_shapes_config = "(T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32))"

class Repro(torch.nn.Module):
    def forward(self, relu_19: "f32[128, 384, 14, 14]", arg322_1: "f32[384]", arg323_1: "f32[384]", arg324_1: "f32[384]", arg325_1: "f32[384]", convolution_40: "f32[128, 384, 14, 14]", arg328_1: "f32[384]", arg329_1: "f32[384]", arg330_1: "f32[384]", arg331_1: "f32[384]", convolution_41: "f32[128, 384, 14, 14]", arg334_1: "f32[384]", arg335_1: "f32[384]", arg336_1: "f32[384]", arg337_1: "f32[384]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(relu_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 384, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_19, getitem_1);  relu_19 = None
        mul_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        squeeze_dims: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_default, [0, 2, 3]);  rsqrt_default = None
        mul_tensor_1: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_2: "f32[384]" = torch.ops.aten.mul.Tensor(arg322_1, 0.9)
        add_tensor_1: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_2: "f32[384]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0000398612827361);  squeeze_dims_2 = None
        mul_tensor_4: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[384]" = torch.ops.aten.mul.Tensor(arg323_1, 0.9)
        add_tensor_2: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_40, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 384, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 384, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_default_1: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_3);  convolution_40 = None
        mul_tensor_7: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        squeeze_dims_3: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_dims_4: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_default_1, [0, 2, 3]);  rsqrt_default_1 = None
        mul_tensor_8: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.1)
        mul_tensor_9: "f32[384]" = torch.ops.aten.mul.Tensor(arg328_1, 0.9)
        add_tensor_5: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        squeeze_dims_5: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_10: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.0000398612827361);  squeeze_dims_5 = None
        mul_tensor_11: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.1);  mul_tensor_10 = None
        mul_tensor_12: "f32[384]" = torch.ops.aten.mul.Tensor(arg329_1, 0.9)
        add_tensor_6: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_4: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_default_5: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_13: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5);  mul_tensor_7 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
        unsqueeze_default_7: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_7: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7);  mul_tensor_13 = unsqueeze_default_7 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 384, 1, 1]" = var_mean_correction_2[0]
        getitem_5: "f32[1, 384, 1, 1]" = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_8: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_default_2: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        sub_tensor_2: "f32[128, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_5);  convolution_41 = None
        mul_tensor_14: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = None
        squeeze_dims_6: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_dims_7: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_default_2, [0, 2, 3]);  rsqrt_default_2 = None
        mul_tensor_15: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 0.1)
        mul_tensor_16: "f32[384]" = torch.ops.aten.mul.Tensor(arg334_1, 0.9)
        add_tensor_9: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_8: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_17: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 1.0000398612827361);  squeeze_dims_8 = None
        mul_tensor_18: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[384]" = torch.ops.aten.mul.Tensor(arg335_1, 0.9)
        add_tensor_10: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
        unsqueeze_default_8: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_default_9: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        mul_tensor_20: "f32[128, 384, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_14, unsqueeze_default_9);  mul_tensor_14 = unsqueeze_default_9 = None
        unsqueeze_default_10: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
        unsqueeze_default_11: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        add_tensor_11: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_20, unsqueeze_default_11);  mul_tensor_20 = unsqueeze_default_11 = None
        add_tensor_12: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_7, add_tensor_11);  add_tensor_7 = add_tensor_11 = None
        add_tensor_13: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_tensor_12, add_tensor_3);  add_tensor_12 = add_tensor_3 = None
        relu_default: "f32[128, 384, 14, 14]" = torch.ops.aten.relu.default(add_tensor_13);  add_tensor_13 = None
        unsqueeze_default_12: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_6, 0);  squeeze_dims_6 = None
        unsqueeze_default_13: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_3, 0);  squeeze_dims_3 = None
        unsqueeze_default_16: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_19: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 2);  unsqueeze_default_18 = None
        unsqueeze_default_20: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 3);  unsqueeze_default_19 = None
        copy__default: "f32[384]" = torch.ops.aten.copy_.default(arg322_1, add_tensor_1);  arg322_1 = add_tensor_1 = None
        copy__default_1: "f32[384]" = torch.ops.aten.copy_.default(arg323_1, add_tensor_2);  arg323_1 = add_tensor_2 = None
        copy__default_2: "f32[384]" = torch.ops.aten.copy_.default(arg328_1, add_tensor_5);  arg328_1 = add_tensor_5 = None
        copy__default_3: "f32[384]" = torch.ops.aten.copy_.default(arg329_1, add_tensor_6);  arg329_1 = add_tensor_6 = None
        copy__default_4: "f32[384]" = torch.ops.aten.copy_.default(arg334_1, add_tensor_9);  arg334_1 = add_tensor_9 = None
        copy__default_5: "f32[384]" = torch.ops.aten.copy_.default(arg335_1, add_tensor_10);  arg335_1 = add_tensor_10 = None
        return (squeeze_dims_1, squeeze_dims_4, squeeze_dims_7, relu_default, unsqueeze_default_14, unsqueeze_default_17, unsqueeze_default_20, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5)



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
