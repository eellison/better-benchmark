"""
Standalone repro captured via capture_hook.
Label: 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_infer_000
Pattern hash: 25aabc59b316
Shape hash: e8a99cae
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
_shapes_config = "(T([2048, 2560], f32), T([16, 128, 2560], f32), T([2560], f32), T([2560], f32), S([16, 128, 2560]), S([2048, 2560]), S([2048, 2560]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_5: "f32[2048, 2560]", add_5: "f32[16, 128, 2560]", arg20_1: "f32[2560]", arg21_1: "f32[2560]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[16, 128, 2560]" = torch.ops.aten.view.default(addmm_5, _shape_param_0);  addmm_5 = _shape_param_0 = None
        add_tensor: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_5, view_default);  add_5 = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[16, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_tensor, arg20_1);  mul_tensor = arg20_1 = None
        add_tensor_2: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg21_1);  mul_tensor_1 = arg21_1 = None
        view_default_1: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[2048, 2560]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
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
