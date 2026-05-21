"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train_000
Pattern hash: 7eb8a9047c3e
Shape hash: f55a873b
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
_shapes_config = "(T([128, 128, 56, 56], f32), T([128], f32), T([128], f32), T([128], f32), T([128], f32), S([128, 8, 7, 8, 7, 128]), S([-1, 7, 7, 128]), S([-1, 49, 128]), S([401408, 128]))"

class Repro(torch.nn.Module):
    def forward(self, convolution: "f32[128, 128, 56, 56]", arg3_1: "f32[128]", arg4_1: "f32[128]", arg5_1: "f32[128]", arg6_1: "f32[128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        permute_default: "f32[128, 56, 56, 128]" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None
        var_mean_correction = torch.ops.aten.var_mean.correction(permute_default, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 56, 56, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 56, 56, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(permute_default, getitem_1);  permute_default = getitem_1 = None
        mul_tensor: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor, arg3_1);  mul_tensor = arg3_1 = None
        add_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg4_1);  mul_tensor_1 = arg4_1 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_1, [3], correction = 0, keepdim = True)
        getitem_2: "f32[128, 56, 56, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[128, 56, 56, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[128, 56, 56, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default_1: "f32[128, 56, 56, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[128, 56, 56, 128]" = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_3);  add_tensor_1 = getitem_3 = None
        mul_tensor_2: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_3: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_tensor_2, arg5_1);  mul_tensor_2 = arg5_1 = None
        add_tensor_3: "f32[128, 56, 56, 128]" = torch.ops.aten.add.Tensor(mul_tensor_3, arg6_1);  mul_tensor_3 = arg6_1 = None
        view_default: "f32[128, 8, 7, 8, 7, 128]" = torch.ops.aten.view.default(add_tensor_3, _shape_param_0);  add_tensor_3 = _shape_param_0 = None
        permute_default_1: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.permute.default(view_default, [0, 1, 3, 2, 4, 5]);  view_default = None
        clone_default: "f32[128, 8, 8, 7, 7, 128]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_1: "f32[8192, 7, 7, 128]" = torch.ops.aten.view.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        view_default_2: "f32[8192, 49, 128]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        view_default_3: "f32[401408, 128]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        return view_default_3



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
