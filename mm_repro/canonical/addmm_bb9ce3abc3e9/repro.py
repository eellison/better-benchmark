"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/BartForCausalLM/full_graph_000.py
Pattern hash: bb9ce3abc3e9
Shape hash: c0f1aec2
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
    def forward(self, arg0_1: "bf16[8, 16, 1024, 64]", arg1_1: "bf16[1024, 1024]", arg2_1: "bf16[1024]", arg3_1: "bf16[8, 1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute: "bf16[8, 1024, 16, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[8, 1024, 1024]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        view_1: "bf16[8192, 1024]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg2_1, view_1, permute_1);  arg2_1 = view_1 = permute_1 = None
        view_2: "bf16[8, 1024, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        add: "bf16[8, 1024, 1024]" = torch.ops.aten.add.Tensor(arg3_1, view_2);  arg3_1 = view_2 = None
        convert_element_type: "f32[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        return convert_element_type



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
