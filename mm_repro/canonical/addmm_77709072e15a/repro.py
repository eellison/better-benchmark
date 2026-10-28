"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/M2M100ForConditionalGeneration/full_graph_000.py
Pattern hash: 77709072e15a
Shape hash: 34d75d82
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
    def forward(self, arg0_1: "bf16[1024, 4096]", arg1_1: "bf16[1024]", arg2_1: "bf16[8192, 4096]", arg3_1: "bf16[1024, 1024]", arg4_1: "bf16[1024]", arg5_1: "bf16[8192, 1024]", arg6_1: "bf16[1024, 1024]", arg7_1: "bf16[1024]", arg8_1: "bf16[8192, 1024]", arg9_1: "bf16[1024, 1024]", arg10_1: "bf16[1024]", arg11_1: "bf16[8192, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6):
        # No stacktrace found for following nodes
        permute: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        addmm: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg1_1, arg2_1, permute);  arg1_1 = arg2_1 = permute = None
        view: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        permute_1: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        addmm_1: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg4_1, arg5_1, permute_1);  arg4_1 = arg5_1 = permute_1 = None
        view_1: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(addmm_1, _shape_param_1);  addmm_1 = _shape_param_1 = None
        view_2: "bf16[64, 128, 16, 64]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute_2: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_3: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm_2: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg7_1, arg8_1, permute_3);  arg7_1 = arg8_1 = permute_3 = None
        view_3: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(addmm_2, _shape_param_3);  addmm_2 = _shape_param_3 = None
        view_4: "bf16[64, 128, 16, 64]" = torch.ops.aten.view.default(view_3, _shape_param_4);  view_3 = _shape_param_4 = None
        permute_4: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        permute_5: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_3: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg10_1, arg11_1, permute_5);  arg10_1 = arg11_1 = permute_5 = None
        view_5: "bf16[64, 128, 1024]" = torch.ops.aten.view.default(addmm_3, _shape_param_5);  addmm_3 = _shape_param_5 = None
        view_6: "bf16[64, 128, 16, 64]" = torch.ops.aten.view.default(view_5, _shape_param_6);  view_5 = _shape_param_6 = None
        permute_6: "bf16[64, 16, 128, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        return (view, permute_2, permute_4, permute_6)



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
