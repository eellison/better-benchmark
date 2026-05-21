"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train_000
Pattern hash: f0d7c08a0622
Shape hash: 2351776b
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
_shapes_config = "(T([128, 64, 147, 147], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_2: "f32[128, 64, 147, 147]", arg15_1: "f32[64]", arg16_1: "f32[64]", arg17_1: "f32[64]", arg18_1: "f32[64]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt_default: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 64, 147, 147]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_1);  convolution_2 = None
        mul_tensor: "f32[128, 64, 147, 147]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(arg15_1, 0.9)
        add_tensor_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_1 = mul_tensor_2 = None
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0000003615393043);  squeeze_dims_1 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.1);  mul_tensor_3 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(arg16_1, 0.9)
        add_tensor_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_default_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_6: "f32[128, 64, 147, 147]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(arg18_1, -1);  arg18_1 = None
        unsqueeze_default_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_3: "f32[128, 64, 147, 147]" = torch.ops.aten.add.Tensor(mul_tensor_6, unsqueeze_default_3);  mul_tensor_6 = unsqueeze_default_3 = None
        relu_default: "f32[128, 64, 147, 147]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_default = None
        getitem_2: "f32[128, 64, 73, 73]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_3: "i8[128, 64, 73, 73]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None
        copy__default: "f32[64]" = torch.ops.aten.copy_.default(arg15_1, add_tensor_1);  arg15_1 = add_tensor_1 = None
        copy__default_1: "f32[64]" = torch.ops.aten.copy_.default(arg16_1, add_tensor_2);  arg16_1 = add_tensor_2 = None
        return (getitem_2, getitem_3, copy__default, copy__default_1)



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
