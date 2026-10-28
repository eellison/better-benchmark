"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/BlenderbotForCausalLM/full_graph_007.py
Pattern hash: d5f7be69a724
Shape hash: a3e8697c
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
    def forward(self, arg0_1: "bf16[32, 32, 128, 80]", arg1_1: "bf16[2560, 4096]", arg2_1: "bf16[32, 32, 128, 80]", arg3_1: "bf16[32, 32, 128, 80]", arg4_1: "bf16[4096, 2560]", arg5_1: "bf16[32, 32, 128, 80]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        permute: "bf16[32, 128, 32, 80]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        view_1: "bf16[4096, 2560]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        mm: "bf16[2560, 2560]" = torch.ops.aten.mm.default(arg1_1, view_1);  arg1_1 = view_1 = None
        convert_element_type: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        permute_1: "bf16[32, 128, 32, 80]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        view_2: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(permute_1, _shape_param_2);  permute_1 = _shape_param_2 = None
        permute_2: "bf16[32, 128, 32, 80]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1, 3]);  arg3_1 = None
        view_3: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(permute_2, _shape_param_3);  permute_2 = _shape_param_3 = None
        view_4: "bf16[4096, 2560]" = torch.ops.aten.view.default(view_2, _shape_param_4);  view_2 = _shape_param_4 = None
        permute_3: "bf16[2560, 4096]" = torch.ops.aten.permute.default(view_4, [1, 0]);  view_4 = None
        mm_1: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_3, arg4_1);  permute_3 = None
        convert_element_type_1: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        view_5: "bf16[4096, 2560]" = torch.ops.aten.view.default(view_3, _shape_param_5);  view_3 = _shape_param_5 = None
        permute_4: "bf16[2560, 4096]" = torch.ops.aten.permute.default(view_5, [1, 0]);  view_5 = None
        mm_2: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_4, arg4_1);  permute_4 = None
        convert_element_type_2: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        permute_5: "bf16[32, 128, 32, 80]" = torch.ops.aten.permute.default(arg5_1, [0, 2, 1, 3]);  arg5_1 = None
        view_6: "bf16[32, 128, 2560]" = torch.ops.aten.view.default(permute_5, _shape_param_6);  permute_5 = _shape_param_6 = None
        view_7: "bf16[4096, 2560]" = torch.ops.aten.view.default(view_6, _shape_param_7);  view_6 = _shape_param_7 = None
        permute_6: "bf16[2560, 4096]" = torch.ops.aten.permute.default(view_7, [1, 0]);  view_7 = None
        mm_3: "bf16[2560, 2560]" = torch.ops.aten.mm.default(permute_6, arg4_1);  permute_6 = arg4_1 = None
        convert_element_type_3: "f32[2560, 2560]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
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
