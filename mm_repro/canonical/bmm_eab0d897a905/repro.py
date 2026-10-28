"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/TrOCRForCausalLM/full_graph_006.py
Pattern hash: eab0d897a905
Shape hash: 7a8d3243
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
    def forward(self, arg0_1: "bf16[1024, 256, 64]", arg1_1: "bf16[1024, 64, 256]", arg2_1: "bf16[1024, 256, 256]", arg3_1: "f32[64, 1, 256, 256]", arg4_1: "f32[1024, 256, 1]", arg5_1: "f32[1024, 256, 1]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        bmm: "bf16[1024, 256, 256]" = torch.ops.aten.bmm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[1024, 256, 256]" = torch.ops.prims.convert_element_type.default(bmm, torch.float32);  bmm = None
        view: "bf16[64, 16, 256, 256]" = torch.ops.aten.view.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        add: "f32[64, 16, 256, 256]" = torch.ops.aten.add.Tensor(view, arg3_1);  view = arg3_1 = None
        view_1: "f32[1024, 256, 256]" = torch.ops.aten.view.default(add, _shape_param_1);  add = _shape_param_1 = None
        sub: "f32[1024, 256, 256]" = torch.ops.aten.sub.Tensor(view_1, arg4_1);  view_1 = arg4_1 = None
        exp: "f32[1024, 256, 256]" = torch.ops.aten.exp.default(sub);  sub = None
        div: "f32[1024, 256, 256]" = torch.ops.aten.div.Tensor(exp, arg5_1);  exp = arg5_1 = None
        mul: "f32[1024, 256, 256]" = torch.ops.aten.mul.Tensor(convert_element_type, div);  convert_element_type = div = None
        return mul



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
