"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/GPTJForCausalLM/full_graph_000.py
Pattern hash: e89dd80d88ee
Shape hash: a5dac7a3
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[128, 4096]", arg2_1: "bf16[4096, 16384]", arg3_1: "bf16[4096]", arg4_1: "bf16[128, 16384]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        permute: "bf16[4096, 4096]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(arg1_1, permute);  arg1_1 = permute = None
        view: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        permute_1: "bf16[16384, 4096]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm: "bf16[128, 4096]" = torch.ops.aten.addmm.default(arg3_1, arg4_1, permute_1);  arg3_1 = arg4_1 = permute_1 = None
        view_1: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        add: "bf16[1, 128, 4096]" = torch.ops.aten.add.Tensor(view, view_1);  view = view_1 = None
        return add



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
