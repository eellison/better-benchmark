"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-8-9-linux.aws.h100_graph74
Pattern hash: 91afe18f34c2
Shape hash: e26cd3bb
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 1280], f64), T([1, 64, 64, 1280], f64, stride=(5242880, 64, 1, 4096)), T([1280], f64), T([1280], f64), T([3840, 1280], f64), S([1, 64, 64, 1280]), S([1, 5, 14, 5, 14, 1280]), S([-1, 14, 14, 1280]), S([4900, 1280]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_119: "f64[4096, 1280]", add_326: "f64[1, 64, 64, 1280]", arg424_1: "f64[1280]", arg425_1: "f64[1280]", arg426_1: "f64[3840, 1280]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view_default: "f64[1, 64, 64, 1280]" = torch.ops.aten.view.default(addmm_119, _shape_param_0);  addmm_119 = _shape_param_0 = None
        add_tensor: "f64[1, 64, 64, 1280]" = torch.ops.aten.add.Tensor(add_326, view_default);  add_326 = view_default = None
        clone_default: "f64[1, 64, 64, 1280]" = torch.ops.aten.clone.default(add_tensor, memory_format = torch.contiguous_format);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(clone_default, [3], correction = 0, keepdim = True)
        getitem: "f64[1, 64, 64, 1]" = var_mean_correction[0]
        getitem_1: "f64[1, 64, 64, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f64[1, 64, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f64[1, 64, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f64[1, 64, 64, 1280]" = torch.ops.aten.sub.Tensor(clone_default, getitem_1);  clone_default = getitem_1 = None
        mul_tensor: "f64[1, 64, 64, 1280]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f64[1, 64, 64, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor, arg424_1);  mul_tensor = arg424_1 = None
        add_tensor_2: "f64[1, 64, 64, 1280]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg425_1);  mul_tensor_1 = arg425_1 = None
        constant_pad_nd_default: "f64[1, 70, 70, 1280]" = torch.ops.aten.constant_pad_nd.default(add_tensor_2, [0, 0, 0, 6, 0, 6], 0.0);  add_tensor_2 = None
        view_default_1: "f64[1, 5, 14, 5, 14, 1280]" = torch.ops.aten.view.default(constant_pad_nd_default, _shape_param_1);  constant_pad_nd_default = _shape_param_1 = None
        permute_default: "f64[1, 5, 5, 14, 14, 1280]" = torch.ops.aten.permute.default(view_default_1, [0, 1, 3, 2, 4, 5]);  view_default_1 = None
        clone_default_1: "f64[1, 5, 5, 14, 14, 1280]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_2: "f64[25, 14, 14, 1280]" = torch.ops.aten.view.default(clone_default_1, _shape_param_2);  clone_default_1 = _shape_param_2 = None
        view_default_3: "f64[4900, 1280]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        permute_default_1: "f64[1280, 3840]" = torch.ops.aten.permute.default(arg426_1, [1, 0]);  arg426_1 = None
        return (view_default_3, permute_default_1)


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
