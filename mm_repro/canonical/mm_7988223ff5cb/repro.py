"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/deit_base_distilled_patch16_224/full_graph_001.py
Pattern hash: 7988223ff5cb
Shape hash: e920b59b
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
    def forward(self, arg0_1: "bf16[128, 1000]", arg1_1: "bf16[128, 768]", arg2_1: "bf16[128, 768]"):
        # No stacktrace found for following nodes
        div: "bf16[128, 1000]" = torch.ops.aten.div.Tensor(arg0_1, 2);  arg0_1 = None
        permute: "bf16[1000, 128]" = torch.ops.aten.permute.default(div, [1, 0]);  div = None
        mm: "bf16[1000, 768]" = torch.ops.aten.mm.default(permute, arg1_1);  arg1_1 = None
        convert_element_type: "f32[1000, 768]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mm_1: "bf16[1000, 768]" = torch.ops.aten.mm.default(permute, arg2_1);  permute = arg2_1 = None
        convert_element_type_1: "f32[1000, 768]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
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
