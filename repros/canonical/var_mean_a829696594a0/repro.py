"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train_000
Pattern hash: a829696594a0
Shape hash: ad9f6940
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([12000, 384], f32), T([8, 1500, 384], f32), T([384], f32), T([384], f32), S([8, 1500, 384]), S([12000, 384]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_2: "f32[12000, 384]", arg2_1: "f32[8, 1500, 384]", arg10_1: "f32[384]", arg11_1: "f32[384]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[8, 1500, 384]" = torch.ops.aten.view.default(addmm_2, _shape_param_0);  addmm_2 = _shape_param_0 = None
        add_tensor: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(arg2_1, view_default);  arg2_1 = view_default = None
        clone_default: "f32[8, 1500, 384]" = torch.ops.aten.clone.default(add_tensor, memory_format = torch.contiguous_format);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(clone_default, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 1500, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 1500, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[8, 1500, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[8, 1500, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[8, 1500, 384]" = torch.ops.aten.sub.Tensor(clone_default, getitem_1);  clone_default = getitem_1 = None
        mul_tensor: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 1500, 384]" = torch.ops.aten.mul.Tensor(mul_tensor, arg10_1);  mul_tensor = arg10_1 = None
        add_tensor_2: "f32[8, 1500, 384]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg11_1);  mul_tensor_1 = arg11_1 = None
        view_default_1: "f32[12000, 384]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  add_tensor_2 = _shape_param_1 = None
        return view_default_1

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
