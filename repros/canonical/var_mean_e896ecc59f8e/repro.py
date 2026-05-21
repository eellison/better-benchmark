"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_infer_000
Pattern hash: e896ecc59f8e
Shape hash: 05d74714
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
_shapes_config = "(T([4096, 4096], f32), T([8, 512, 4096], f32), T([4096], f32), T([4096], f32), S([8, 512, 4096]), S([4096, 4096]), S([4096, 4096]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_54: "f32[4096, 4096]", add_80: "f32[8, 512, 4096]", arg24_1: "f32[4096]", arg25_1: "f32[4096]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[8, 512, 4096]" = torch.ops.aten.view.default(addmm_54, _shape_param_0);  addmm_54 = _shape_param_0 = None
        add_tensor: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(view_default, add_80);  view_default = add_80 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[8, 512, 4096]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_tensor, arg24_1);  mul_tensor = arg24_1 = None
        add_tensor_2: "f32[8, 512, 4096]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg25_1);  mul_tensor_1 = arg25_1 = None
        view_default_1: "f32[4096, 4096]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[4096, 4096]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        return (view_default_1, view_default_2)



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
