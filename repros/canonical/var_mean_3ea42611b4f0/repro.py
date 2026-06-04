"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_CycleGAN_and_pix2pix_infer_000
Pattern hash: 3ea42611b4f0
Shape hash: 15254c9a
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([1, 256, 64, 64], f32), T([1, 256, 64, 64], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_18: "f32[1, 256, 64, 64]", add_23: "f32[1, 256, 64, 64]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 256, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 256, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_1);  convolution_18 = getitem_1 = None
        add_tensor: "f32[1, 256, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 256, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        add_tensor_1: "f32[1, 256, 64, 64]" = torch.ops.aten.add.Tensor(add_23, mul_tensor);  add_23 = mul_tensor = None
        iota_default: "i64[66]" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_default: "i64[66]" = torch.ops.aten.abs.default(iota_default);  iota_default = None
        sub_tensor_1: "i64[66]" = torch.ops.aten.sub.Tensor(63, abs_default);  abs_default = None
        abs_default_1: "i64[66]" = torch.ops.aten.abs.default(sub_tensor_1);  sub_tensor_1 = None
        sub_tensor_2: "i64[66]" = torch.ops.aten.sub.Tensor(63, abs_default_1);  abs_default_1 = None
        _unsafe_index_tensor: "f32[1, 256, 66, 64]" = torch.ops.aten._unsafe_index.Tensor(add_tensor_1, [None, None, sub_tensor_2, None]);  add_tensor_1 = sub_tensor_2 = None
        iota_default_1: "i64[66]" = torch.ops.prims.iota.default(66, start = -1, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_default_2: "i64[66]" = torch.ops.aten.abs.default(iota_default_1);  iota_default_1 = None
        sub_tensor_3: "i64[66]" = torch.ops.aten.sub.Tensor(63, abs_default_2);  abs_default_2 = None
        abs_default_3: "i64[66]" = torch.ops.aten.abs.default(sub_tensor_3);  sub_tensor_3 = None
        sub_tensor_4: "i64[66]" = torch.ops.aten.sub.Tensor(63, abs_default_3);  abs_default_3 = None
        _unsafe_index_tensor_1: "f32[1, 256, 66, 66]" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_tensor, [None, None, None, sub_tensor_4]);  _unsafe_index_tensor = sub_tensor_4 = None
        return _unsafe_index_tensor_1

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
