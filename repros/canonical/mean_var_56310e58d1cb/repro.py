"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_infer_000
Pattern hash: 56310e58d1cb
Shape hash: da5ecc03
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
_shapes_config = "(T([16384, 768], f32), T([128, 128, 768], f32), T([768], f32), T([768], f32), S([128, 128, 768]), S([16384, 768]), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_65: "f32[16384, 768]", add_74: "f32[128, 128, 768]", arg181_1: "f32[768]", arg182_1: "f32[768]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_65, _shape_param_0);  addmm_65 = _shape_param_0 = None
        add_tensor: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_74, view_default);  add_74 = view_default = None
        mean_dim: "f32[128, 128, 1]" = torch.ops.aten.mean.dim(add_tensor, [-1], True)
        sub_tensor: "f32[128, 128, 768]" = torch.ops.aten.sub.Tensor(add_tensor, mean_dim);  mean_dim = None
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(arg181_1, sub_tensor);  arg181_1 = sub_tensor = None
        var_correction: "f32[128, 128, 1]" = torch.ops.aten.var.correction(add_tensor, [-1], correction = 1.0, keepdim = True);  add_tensor = None
        sqrt_default: "f32[128, 128, 1]" = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        add_tensor_1: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_default, 1e-06);  sqrt_default = None
        div_tensor: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_1);  mul_tensor = add_tensor_1 = None
        add_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(div_tensor, arg182_1);  div_tensor = arg182_1 = None
        view_default_1: "f32[16384, 768]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[16384, 768]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
        return (view_default_2, view_default_1)



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
