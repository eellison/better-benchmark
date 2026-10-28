"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/AlbertForMaskedLM/full_graph_001.py
Pattern hash: 5e7f964ce981
Shape hash: a1fc3fc7
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
    def forward(self, arg0_1: "bf16[4096, 4096]", arg1_1: "bf16[4096, 16384]", arg2_1: "bf16[4096, 4096]", arg3_1: "bf16[4096, 16384]", arg4_1: "bf16[4096, 4096]", arg5_1: "bf16[4096, 16384]", arg6_1: "bf16[4096, 4096]", arg7_1: "bf16[4096, 16384]"):
        # No stacktrace found for following nodes
        mm: "bf16[4096, 16384]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mm_1: "bf16[4096, 16384]" = torch.ops.aten.mm.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        convert_element_type_1: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        add: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(convert_element_type, convert_element_type_1);  convert_element_type = convert_element_type_1 = None
        mm_2: "bf16[4096, 16384]" = torch.ops.aten.mm.default(arg4_1, arg5_1);  arg4_1 = arg5_1 = None
        convert_element_type_2: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        add_1: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add, convert_element_type_2);  add = convert_element_type_2 = None
        mm_3: "bf16[4096, 16384]" = torch.ops.aten.mm.default(arg6_1, arg7_1);  arg6_1 = arg7_1 = None
        convert_element_type_3: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        add_2: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_3);  add_1 = convert_element_type_3 = None
        return add_2



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
