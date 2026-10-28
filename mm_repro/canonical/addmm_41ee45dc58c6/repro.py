"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/deit_base_distilled_patch16_224/full_graph_000.py
Pattern hash: 41ee45dc58c6
Shape hash: 96377bb5
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
    def forward(self, arg0_1: "f32[1000]", arg1_1: "f32[1000, 768]", arg2_1: "bf16[128, 768]", arg3_1: "f32[1000]", arg4_1: "f32[1000, 768]", arg5_1: "bf16[128, 768]"):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[1000, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        permute: "bf16[768, 1000]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[128, 1000]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, permute);  convert_element_type = arg2_1 = permute = None
        convert_element_type_2: "bf16[1000]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_3: "bf16[1000, 768]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        permute_1: "bf16[768, 1000]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        addmm_1: "bf16[128, 1000]" = torch.ops.aten.addmm.default(convert_element_type_2, arg5_1, permute_1);  convert_element_type_2 = arg5_1 = permute_1 = None
        add: "bf16[128, 1000]" = torch.ops.aten.add.Tensor(addmm, addmm_1);  addmm = addmm_1 = None
        div: "bf16[128, 1000]" = torch.ops.aten.div.Tensor(add, 2);  add = None
        return div



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
