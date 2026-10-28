"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/BlenderbotForConditionalGeneration/full_graph_013.py
Pattern hash: b295495ff6a1
Shape hash: 8549c0bb
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
    def forward(self, arg0_1: "bf16[2560, 2048]", arg1_1: "bf16[2048, 2560]", arg2_1: "bf16[2560, 2048]", arg3_1: "bf16[2560, 2048]"):
        # No stacktrace found for following nodes
        mm: "bf16[2560, 2560]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = None
        convert_element_type: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mm_1: "bf16[2560, 2560]" = torch.ops.aten.mm.default(arg2_1, arg1_1);  arg2_1 = None
        convert_element_type_1: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        mm_2: "bf16[2560, 2560]" = torch.ops.aten.mm.default(arg3_1, arg1_1);  arg3_1 = arg1_1 = None
        convert_element_type_2: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        return (convert_element_type, convert_element_type_1, convert_element_type_2)



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
