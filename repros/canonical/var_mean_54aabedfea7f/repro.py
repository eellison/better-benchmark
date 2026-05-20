"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-8-9-linux.aws.h100_graph74
Pattern hash: 54aabedfea7f
Shape hash: adf78bab
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 1280], f64), T([1, 64, 64, 1280], f64, stride=(5242880, 64, 1, 4096)), T([1280], f64), T([1280], f64), T([5120, 1280], f64), S([1, 64, 64, 1280]), S([4096, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_125: "f64[4096, 1280]", add_341: "f64[1, 64, 64, 1280]", arg446_1: "f64[1280]", arg447_1: "f64[1280]", arg448_1: "f64[5120, 1280]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f64[1, 64, 64, 1280]" = torch.ops.aten.view.default(addmm_125, _shape_param_0);  addmm_125 = _shape_param_0 = None
        add_tensor: "f64[1, 64, 64, 1280]" = torch.ops.aten.add.Tensor(add_341, view_default);  add_341 = view_default = None
        clone_default: "f64[1, 64, 64, 1280]" = torch.ops.aten.clone.default(add_tensor, memory_format = torch.contiguous_format);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(clone_default, [3], correction = 0, keepdim = True)
        getitem: "f64[1, 64, 64, 1]" = var_mean_correction[0]
        getitem_1: "f64[1, 64, 64, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f64[1, 64, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f64[1, 64, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f64[1, 64, 64, 1280]" = torch.ops.aten.sub.Tensor(clone_default, getitem_1);  clone_default = getitem_1 = None
        mul_tensor: "f64[1, 64, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f64[1, 64, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor, arg446_1);  mul_tensor = arg446_1 = None
        add_tensor_2: "f64[1, 64, 64, 1280]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg447_1);  mul_tensor_1 = arg447_1 = None
        view_default_1: "f64[4096, 1280]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        permute_default: "f64[1280, 5120]" = torch.ops.aten.permute.default(arg448_1, [1, 0]);  arg448_1 = None
        return (view_default_1, permute_default)


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
