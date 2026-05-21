"""
Standalone repro captured via capture_hook.
Label: torchbench_pytorch_CycleGAN_and_pix2pix_infer_000
Pattern hash: 58cdb69fc823
Shape hash: b39a61e1
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
_shapes_config = "(T([1, 64, 256, 256], f32))"

class Repro(torch.nn.Module):
    def forward(self, convolution_22: "f32[1, 64, 256, 256]"):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 64, 256, 256]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_1);  convolution_22 = getitem_1 = None
        add_tensor: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 64, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        relu_default: "f32[1, 64, 256, 256]" = torch.ops.aten.relu.default(mul_tensor);  mul_tensor = None
        iota_default: "i64[262]" = torch.ops.prims.iota.default(262, start = -3, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_default: "i64[262]" = torch.ops.aten.abs.default(iota_default);  iota_default = None
        sub_tensor_1: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_default);  abs_default = None
        abs_default_1: "i64[262]" = torch.ops.aten.abs.default(sub_tensor_1);  sub_tensor_1 = None
        sub_tensor_2: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_default_1);  abs_default_1 = None
        _unsafe_index_tensor: "f32[1, 64, 262, 256]" = torch.ops.aten._unsafe_index.Tensor(relu_default, [None, None, sub_tensor_2, None]);  relu_default = sub_tensor_2 = None
        iota_default_1: "i64[262]" = torch.ops.prims.iota.default(262, start = -3, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        abs_default_2: "i64[262]" = torch.ops.aten.abs.default(iota_default_1);  iota_default_1 = None
        sub_tensor_3: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_default_2);  abs_default_2 = None
        abs_default_3: "i64[262]" = torch.ops.aten.abs.default(sub_tensor_3);  sub_tensor_3 = None
        sub_tensor_4: "i64[262]" = torch.ops.aten.sub.Tensor(255, abs_default_3);  abs_default_3 = None
        _unsafe_index_tensor_1: "f32[1, 64, 262, 262]" = torch.ops.aten._unsafe_index.Tensor(_unsafe_index_tensor, [None, None, None, sub_tensor_4]);  _unsafe_index_tensor = sub_tensor_4 = None
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
