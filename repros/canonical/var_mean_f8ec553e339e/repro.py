"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_train
Pattern hash: f8ec553e339e
Shape hash: 23d9c217
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 128, 2560]", arg1_1: "f32[2560]", arg2_1: "f32[2560]", _shape_param_0):
        # No stacktrace found for following nodes
        var_mean = torch.ops.aten.var_mean.correction(arg0_1, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean[0]
        getitem_1: "f32[32, 128, 1]" = var_mean[1];  var_mean = None
        add: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        sub: "f32[32, 128, 2560]" = torch.ops.aten.sub.Tensor(arg0_1, getitem_1);  arg0_1 = None
        mul: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[32, 128, 2560]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        convert_element_type: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        view: "bf16[4096, 2560]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        return (getitem_1, rsqrt, view)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
