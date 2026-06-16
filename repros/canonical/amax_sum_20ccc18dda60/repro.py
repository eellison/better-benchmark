"""
Standalone repro captured via capture_hook.
Label: hf_TrOCRForCausalLM_infer
Pattern hash: 20ccc18dda60
Shape hash: 38831ee4
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
    def forward(self, arg0_1: "bf16[1024, 256, 256]", arg1_1: "bf16[64, 1, 256, 256]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[64, 16, 256, 256]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        add: "bf16[64, 16, 256, 256]" = torch.ops.aten.add.Tensor(view, arg1_1);  view = arg1_1 = None
        view_1: "bf16[1024, 256, 256]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        convert_element_type: "f32[1024, 256, 256]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        amax: "f32[1024, 256, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[1024, 256, 256]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[1024, 256, 256]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[1024, 256, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[1024, 256, 256]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[1024, 256, 256]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        return convert_element_type_1



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
