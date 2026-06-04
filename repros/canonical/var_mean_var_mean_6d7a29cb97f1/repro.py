"""
Standalone repro captured via capture_hook.
Label: torchbench_densenet121_train_000
Pattern hash: 6d7a29cb97f1
Shape hash: 78bba510
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([64, 64, 112, 112], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution: "f32[64, 64, 112, 112]", arg3_1: "f32[64]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[64]", arg8_1: "f32[64]", arg9_1: "f32[64]", arg10_1: "f32[64]", arg11_1: "f32[64]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt_default: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[64, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1);  convolution = None
        mul_tensor: "f32[64, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(arg3_1, 0.9)
        add_tensor_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0000012456169853);  squeeze_dims_1 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(arg4_1, 0.9)
        add_tensor_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[64, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, -1);  arg6_1 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[64, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[64, 64, 112, 112]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu_default = None
        getitem_2: "f32[64, 64, 56, 56]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_3: "i8[64, 64, 56, 56]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(getitem_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 64, 1, 1]" = var_mean_correction_1[0]
        getitem_5: "f32[1, 64, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_default_1: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1: "f32[64, 64, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_2, getitem_5)
        mul_tensor_7: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        squeeze_dims_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_dims_3: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_default_1, [0, 2, 3]);  rsqrt_default_1 = None
        mul_tensor_8: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 0.1)
        mul_tensor_9: "f32[64]" = torch.ops.aten.mul.Tensor(arg8_1, 0.9)
        add_tensor_5: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        squeeze_dims_4: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_10: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_4, 1.0000049824865598);  squeeze_dims_4 = None
        mul_tensor_11: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.1);  mul_tensor_10 = None
        mul_tensor_12: "f32[64]" = torch.ops.aten.mul.Tensor(arg9_1, 0.9)
        add_tensor_6: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_12);  mul_tensor_11 = mul_tensor_12 = None
        unsqueeze_default_4: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_default_5: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_13: "f32[64, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor_7, unsqueeze_default_5);  mul_tensor_7 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_default_7: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_7: "f32[64, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_13, unsqueeze_default_7);  mul_tensor_13 = unsqueeze_default_7 = None
        relu_default_1: "f32[64, 64, 56, 56]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None
        unsqueeze_default_8: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_9: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 3);  unsqueeze_default_9 = None
        sub_tensor_2: "f32[64, 64, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_2, unsqueeze_default_10);  getitem_2 = unsqueeze_default_10 = None
        copy__default: "f32[64]" = torch.ops.aten.copy_.default(arg3_1, add_tensor_1);  arg3_1 = add_tensor_1 = None
        copy__default_1: "f32[64]" = torch.ops.aten.copy_.default(arg4_1, add_tensor_2);  arg4_1 = add_tensor_2 = None
        copy__default_2: "f32[64]" = torch.ops.aten.copy_.default(arg8_1, add_tensor_5);  arg8_1 = add_tensor_5 = None
        copy__default_3: "f32[64]" = torch.ops.aten.copy_.default(arg9_1, add_tensor_6);  arg9_1 = add_tensor_6 = None
        return (getitem_3, squeeze_dims_3, relu_default_1, sub_tensor_2, copy__default, copy__default_1, copy__default_2, copy__default_3)

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
