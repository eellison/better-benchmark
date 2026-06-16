"""
Standalone repro captured via capture_hook.
Label: hf_GoogleFnet_infer
Pattern hash: e45f42fbfc2b
Shape hash: 98ade792
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
    def forward(self, arg0_1: "f32[32, 512, 768, 2]", arg1_1: "f32[32, 512, 768]", arg2_1: "f32[768]", arg3_1: "f32[768]", _shape_param_0):
        # No stacktrace found for following nodes
        select: "f32[32, 512, 768]" = torch.ops.aten.select.int(arg0_1, 3, 0);  arg0_1 = None
        add: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(arg1_1, select);  arg1_1 = select = None
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 512, 1]" = var_mean[0]
        getitem_1: "f32[32, 512, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        add_1: "f32[32, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[32, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
        add_2: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        view: "f32[16384, 768]" = torch.ops.aten.view.default(add_2, _shape_param_0);  _shape_param_0 = None
        return (add_2, view)



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
