"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train_000
Pattern hash: 08f687858162
Shape hash: d21da7a9
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
_shapes_config = "(T([128, 640, 7, 7], f32), T([128, 640, 7, 7], f32), T([640], f32), T([640], f32), S([128, 640]))"

class Repro(torch.nn.Module):
    def forward(self, convolution_45: "f32[128, 640, 7, 7]", add_85: "f32[128, 640, 7, 7]", arg157_1: "f32[640]", arg158_1: "f32[640]", _shape_param_0):
        # No stacktrace found for following nodes
        add_tensor: "f32[128, 640, 7, 7]" = torch.ops.aten.add.Tensor(convolution_45, add_85);  convolution_45 = add_85 = None
        mean_dim: "f32[128, 640, 1, 1]" = torch.ops.aten.mean.dim(add_tensor, [-1, -2], True);  add_tensor = None
        as_strided_default: "f32[128, 640, 1, 1]" = torch.ops.aten.as_strided.default(mean_dim, [128, 640, 1, 1], [640, 1, 640, 640]);  mean_dim = None
        permute_default: "f32[128, 1, 1, 640]" = torch.ops.aten.permute.default(as_strided_default, [0, 2, 3, 1]);  as_strided_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(permute_default, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 1, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 1, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[128, 1, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[128, 1, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[128, 1, 1, 640]" = torch.ops.aten.sub.Tensor(permute_default, getitem_1);  permute_default = getitem_1 = None
        mul_tensor: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(mul_tensor, arg157_1);  mul_tensor = arg157_1 = None
        add_tensor_2: "f32[128, 1, 1, 640]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg158_1);  mul_tensor_1 = arg158_1 = None
        permute_default_1: "f32[128, 640, 1, 1]" = torch.ops.aten.permute.default(add_tensor_2, [0, 3, 1, 2]);  add_tensor_2 = None
        view_default: "f32[128, 640]" = torch.ops.aten.view.default(permute_default_1, _shape_param_0);  permute_default_1 = _shape_param_0 = None
        div_tensor: "f32[128, 1, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_default, 640);  rsqrt_default = None
        return (div_tensor, view_default)



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
