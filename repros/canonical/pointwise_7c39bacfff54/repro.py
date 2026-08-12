"""
Standalone repro captured via capture_hook.
Label: hf_Qwen/Qwen3-0.6B_infer
Pattern hash: 7c39bacfff54
Shape hash: 22f70084
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
    def forward(self, arg0_1: "bf16[1000, 3072]", arg1_1: "bf16[1000, 3072]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[1, 1000, 3072]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[1, 1000, 3072]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        neg: "f32[1, 1000, 3072]" = torch.ops.aten.neg.default(convert_element_type)
        exp: "f32[1, 1000, 3072]" = torch.ops.aten.exp.default(neg);  neg = None
        add: "f32[1, 1000, 3072]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        div: "f32[1, 1000, 3072]" = torch.ops.aten.div.Tensor(convert_element_type, add);  convert_element_type = add = None
        convert_element_type_1: "bf16[1, 1000, 3072]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view_1: "bf16[1, 1000, 3072]" = torch.ops.aten.view.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        mul: "bf16[1, 1000, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_1, view_1);  convert_element_type_1 = view_1 = None
        view_2: "bf16[1000, 3072]" = torch.ops.aten.view.default(mul, _shape_param_2);  mul = _shape_param_2 = None
        return view_2



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
