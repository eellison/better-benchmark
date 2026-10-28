"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/XLNetLMHeadModel/full_graph_001.py
Pattern hash: 0fc65bb383f0
Shape hash: d1284ac9
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
    def forward(self, arg0_1: "bf16[1024, 16384]", arg1_1: "bf16[16384, 1024]", arg2_1: "bf16[1024, 8192]", arg3_1: "bf16[8192, 1024]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        unsqueeze: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm, 0);  mm = None
        view: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(unsqueeze, _shape_param_0);  unsqueeze = _shape_param_0 = None
        squeeze: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(view, 4);  view = None
        squeeze_1: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze, 3);  squeeze = None
        convert_element_type: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_1, torch.float32);  squeeze_1 = None
        mm_1: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        unsqueeze_1: "bf16[1, 1024, 1024]" = torch.ops.aten.unsqueeze.default(mm_1, 0);  mm_1 = None
        view_1: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_1);  unsqueeze_1 = _shape_param_1 = None
        squeeze_2: "bf16[1024, 16, 64, 1]" = torch.ops.aten.squeeze.dim(view_1, 4);  view_1 = None
        squeeze_3: "bf16[1024, 16, 64]" = torch.ops.aten.squeeze.dim(squeeze_2, 3);  squeeze_2 = None
        convert_element_type_1: "f32[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(squeeze_3, torch.float32);  squeeze_3 = None
        return (convert_element_type, convert_element_type_1)



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
