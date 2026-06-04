"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train_004
Pattern hash: dd5260d15336
Shape hash: 3d05ebea
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([8, 4096, 256], f32), T([256], f32), T([256], f32), S([32768, 256]))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "f32[8, 4096, 256]", arg0_1: "f32[256]", arg1_1: "f32[256]", _shape_param_0):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 4096, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 4096, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 4096, 256]" = torch.ops.aten.sub.Tensor(arg2_1, getitem_1);  arg2_1 = getitem_1 = None
        add_tensor: "f32[8, 4096, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[8, 4096, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(mul_tensor, arg0_1);  mul_tensor = arg0_1 = None
        add_tensor_1: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg1_1);  mul_tensor_1 = arg1_1 = None
        view_default: "f32[32768, 256]" = torch.ops.aten.view.default(add_tensor_1, _shape_param_0);  add_tensor_1 = _shape_param_0 = None
        return view_default

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
