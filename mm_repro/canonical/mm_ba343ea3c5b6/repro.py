"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/AlbertForMaskedLM/full_graph_001.py
Pattern hash: ba343ea3c5b6
Shape hash: 6463a5ea
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[4096, 4096]", arg2_1: "bf16[4096, 4096]", arg3_1: "bf16[4096, 4096]", arg4_1: "bf16[4096, 4096]", arg5_1: "bf16[4096, 4096]", arg6_1: "bf16[4096, 4096]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 4096]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mm_1: "bf16[4096, 4096]" = torch.ops.aten.mm.default(arg2_1, arg3_1);  arg2_1 = None
        view: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        convert_element_type_1: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        mm_2: "bf16[4096, 4096]" = torch.ops.aten.mm.default(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        mm_3: "bf16[4096, 4096]" = torch.ops.aten.mm.default(arg6_1, arg3_1);  arg6_1 = arg3_1 = None
        view_1: "bf16[8, 512, 4096]" = torch.ops.aten.view.default(mm_2, _shape_param_1);  mm_2 = _shape_param_1 = None
        convert_element_type_2: "f32[8, 512, 4096]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        convert_element_type_3: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        return (convert_element_type, convert_element_type_1, convert_element_type_2, convert_element_type_3)



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
