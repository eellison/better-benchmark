"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_infer_000
Pattern hash: fb6b847623d0
Shape hash: 312cd8aa
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
_shapes_config = "(T([4096, 2048], f32), T([32, 128, 2048], f32), T([2048], f32), T([2048], f32), S([32, 128, 2048]), S([4096, 2048]), S([4096, 2048]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_68: "f32[4096, 2048]", add_205: "f32[32, 128, 2048]", arg325_1: "f32[2048]", arg326_1: "f32[2048]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "f32[32, 128, 2048]" = torch.ops.aten.view.default(addmm_68, _shape_param_0);  addmm_68 = _shape_param_0 = None
        add_tensor: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_205, view_default);  add_205 = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[32, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        sub_tensor: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        add_tensor_1: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_tensor, arg325_1);  mul_tensor = arg325_1 = None
        add_tensor_2: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg326_1);  mul_tensor_1 = arg326_1 = None
        view_default_1: "f32[4096, 2048]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_1);  _shape_param_1 = None
        view_default_2: "f32[4096, 2048]" = torch.ops.aten.view.default(add_tensor_2, _shape_param_2);  add_tensor_2 = _shape_param_2 = None
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
