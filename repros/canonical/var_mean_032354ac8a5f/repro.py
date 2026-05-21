"""
Standalone repro captured via capture_hook.
Label: torchbench_nanogpt_infer_000
Pattern hash: 032354ac8a5f
Shape hash: f9bbaf9b
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
_shapes_config = "(T([64, 768], f32), T([1, 64, 768], f32), T([768], f32), T([768], f32), S([1, 64, 768]), S([1, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_47: "f32[64, 768]", add_91: "f32[1, 64, 768]", arg147_1: "f32[768]", arg148_1: "f32[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[1, 64, 768]" = torch.ops.aten.view.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None
        add_tensor: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(add_91, view_default);  add_91 = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 64, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 64, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[1, 64, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[1, 64, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 64, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 64, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg147_1);  mul_tensor = arg147_1 = None
        add_tensor_2: "f32[1, 64, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg148_1);  mul_tensor_1 = arg148_1 = None
        full_default: "i64[1]" = torch.ops.aten.full.default([1], -1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_tensor: "f32[1, 1, 768]" = torch.ops.aten.index.Tensor(add_tensor_2, [None, full_default]);  add_tensor_2 = full_default = None
        view_default_1: "f32[1, 768]" = torch.ops.aten.view.default(index_tensor, _shape_param_1);  index_tensor = _shape_param_1 = None
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
