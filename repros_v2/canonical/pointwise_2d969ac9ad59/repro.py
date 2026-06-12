"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_infer
Pattern hash: 2d969ac9ad59
Shape hash: cd47785e
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
    def forward(self, arg0_1: "bf16[4096, 16384]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[8, 512, 16384]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(view, 0.5)
        pow_1: "bf16[8, 512, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view, 3.0)
        mul_1: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(view, mul_1);  view = mul_1 = None
        mul_2: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "bf16[8, 512, 16384]" = torch.ops.aten.tanh.default(mul_2);  mul_2 = None
        add_1: "bf16[8, 512, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
        mul_3: "bf16[8, 512, 16384]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        view_1: "bf16[4096, 16384]" = torch.ops.aten.view.default(mul_3, _shape_param_1);  mul_3 = _shape_param_1 = None
        return view_1



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
