"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-8-9-linux.aws.h100_graph74
Pattern hash: 830747c29ac0
Shape hash: 34389031
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution: "f64[1, 1280, 64, 64]", arg3_1: "f64[1, 64, 64, 1280]", arg4_1: "f64[1280]", arg5_1: "f64[1280]", arg6_1: "f64[3840, 1280]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute_default: "f64[1, 64, 64, 1280]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
        add_tensor: "f64[1, 64, 64, 1280]" = torch.ops.aten.add.Tensor(permute_default, arg3_1);  permute_default = arg3_1 = None
        clone_default: "f64[1, 64, 64, 1280]" = torch.ops.aten.clone.default(add_tensor, memory_format = torch.contiguous_format);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(clone_default, [3], correction = 0, keepdim = True)
        getitem: "f64[1, 64, 64, 1]" = var_mean_correction[0]
        getitem_1: "f64[1, 64, 64, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f64[1, 64, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f64[1, 64, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f64[1, 64, 64, 1280]" = torch.ops.aten.sub.Tensor(clone_default, getitem_1);  clone_default = getitem_1 = None
        mul_tensor: "f64[1, 64, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f64[1, 64, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor, arg4_1);  mul_tensor = arg4_1 = None
        add_tensor_2: "f64[1, 64, 64, 1280]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg5_1);  mul_tensor_1 = arg5_1 = None
        constant_pad_nd_default: "f64[1, 70, 70, 1280]" = torch.ops.aten.constant_pad_nd.default(add_tensor_2, [0, 0, 0, 6, 0, 6], 0.0);  add_tensor_2 = None
        view_default: "f64[1, 5, 14, 5, 14, 1280]" = torch.ops.aten.view.default(constant_pad_nd_default, _shape_param_0);  constant_pad_nd_default = _shape_param_0 = None
        permute_default_1: "f64[1, 5, 5, 14, 14, 1280]" = torch.ops.aten.permute.default(view_default, [0, 1, 3, 2, 4, 5]);  view_default = None
        clone_default_1: "f64[1, 5, 5, 14, 14, 1280]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_1: "f64[25, 14, 14, 1280]" = torch.ops.aten.view.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        view_default_2: "f64[4900, 1280]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default_2: "f64[1280, 3840]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        return (view_default_2, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([1, 1280, 64, 64], dtype=torch.float64, device='cuda'),
    torch.randn([1, 64, 64, 1280], dtype=torch.float64, device='cuda'),
    torch.randn([1280], dtype=torch.float64, device='cuda'),
    torch.randn([1280], dtype=torch.float64, device='cuda'),
    torch.randn([3840, 1280], dtype=torch.float64, device='cuda'),
    [1, 5, 14, 5, 14, 1280],  # _shape_param_0
    [-1, 14, 14, 1280],  # _shape_param_1
    [4900, 1280],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
