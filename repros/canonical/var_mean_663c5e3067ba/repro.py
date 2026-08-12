"""
Standalone repro captured via capture_hook.
Label: torchbench_infer_opacus_cifar10_full_graph_001
Pattern hash: 663c5e3067ba
Shape hash: 7395c806
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
    def forward(self, arg0_1: "f32[64, 64, 16, 16]", arg1_1: "f32[64]", arg2_1: "f32[64]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "f32[64, 32, 2, 256]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        var_mean = torch.ops.aten.var_mean.correction(view, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[64, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[64, 32, 1, 1]" = var_mean[1];  var_mean = None
        sub: "f32[64, 32, 2, 256]" = torch.ops.aten.sub.Tensor(view, getitem_1);  view = getitem_1 = None
        add: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[64, 32, 2, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        view_1: "f32[64, 64, 16, 16]" = torch.ops.aten.view.default(mul, _shape_param_1);  mul = _shape_param_1 = None
        unsqueeze: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
        unsqueeze_1: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        mul_1: "f32[64, 64, 16, 16]" = torch.ops.aten.mul.Tensor(view_1, unsqueeze_2);  view_1 = unsqueeze_2 = None
        unsqueeze_3: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(arg2_1, 0);  arg2_1 = None
        unsqueeze_4: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        unsqueeze_5: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None
        add_1: "f32[64, 64, 16, 16]" = torch.ops.aten.add.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        return add_1



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
