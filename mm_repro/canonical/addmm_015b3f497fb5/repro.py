"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/infer/beit_base_patch16_224/full_graph_000.py
Pattern hash: 015b3f497fb5
Shape hash: 81349049
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
    def forward(self, arg0_1: "bf16[768]", arg1_1: "bf16[768]", arg2_1: "bf16[768]", arg3_1: "bf16[2304, 768]", arg4_1: "bf16[25216, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        cat: "bf16[2304]" = torch.ops.aten.cat.default([arg0_1, arg1_1, arg2_1]);  arg0_1 = arg1_1 = arg2_1 = None
        permute: "bf16[768, 2304]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm: "bf16[25216, 2304]" = torch.ops.aten.addmm.default(cat, arg4_1, permute);  cat = arg4_1 = permute = None
        view: "bf16[128, 197, 2304]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[128, 197, 3, 12, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[3, 128, 12, 197, 64]" = torch.ops.aten.permute.default(view_1, [2, 0, 3, 1, 4]);  view_1 = None
        unbind = torch.ops.aten.unbind.int(permute_1);  permute_1 = None
        getitem: "bf16[128, 12, 197, 64]" = unbind[0]
        getitem_1: "bf16[128, 12, 197, 64]" = unbind[1]
        getitem_2: "bf16[128, 12, 197, 64]" = unbind[2];  unbind = None
        return (getitem, getitem_1, getitem_2)



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
