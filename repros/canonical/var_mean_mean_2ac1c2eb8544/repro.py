"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer_000
Pattern hash: 2ac1c2eb8544
Shape hash: 1dd34346
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
_shapes_config = "(T([6272, 1024], f32), T([128, 49, 1024], f32), T([1024], f32), T([1024], f32), S([128, 49, 1024]), S([128, 7, 7, 1024]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_95: "f32[6272, 1024]", view_650: "f32[128, 49, 1024]", arg361_1: "f32[1024]", arg362_1: "f32[1024]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 49, 1024]" = torch.ops.aten.view.default(addmm_95, _shape_param_0);  addmm_95 = _shape_param_0 = None
        add_tensor: "f32[128, 49, 1024]" = torch.ops.aten.add.Tensor(view_650, view_default);  view_650 = view_default = None
        view_default_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.view.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(view_default_1, [3], correction = 0, keepdim = True)
        getitem: "f32[128, 7, 7, 1]" = var_mean_correction[0]
        getitem_1: "f32[128, 7, 7, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(view_default_1, getitem_1);  view_default_1 = getitem_1 = None
        add_tensor_1: "f32[128, 7, 7, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[128, 7, 7, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg361_1);  mul_tensor = arg361_1 = None
        add_tensor_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg362_1);  mul_tensor_1 = arg362_1 = None
        mean_dim: "f32[128, 1024]" = torch.ops.aten.mean.dim(add_tensor_2, [1, 2]);  add_tensor_2 = None
        return mean_dim



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
