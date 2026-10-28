"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/MegatronBertForCausalLM/full_graph_001.py
Pattern hash: 8e49d8639fb2
Shape hash: 72bac7d3
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
    def forward(self, arg0_1: "bf16[1024, 8192]", arg1_1: "bf16[8192, 1024]", arg2_1: "bf16[16, 16, 512, 64]", arg3_1: "bf16[16, 16, 512, 64]", arg4_1: "bf16[16, 16, 512, 64]", arg5_1: "bf16[8192, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        mm: "bf16[1024, 1024]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        convert_element_type: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        permute: "bf16[16, 512, 16, 64]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        permute_1: "bf16[16, 512, 16, 64]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1, 3]);  arg3_1 = None
        permute_2: "bf16[16, 512, 16, 64]" = torch.ops.aten.permute.default(arg4_1, [0, 2, 1, 3]);  arg4_1 = None
        clone: "bf16[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        view: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        view_1: "bf16[8192, 1024]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_3: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_1, [1, 0]);  view_1 = None
        mm_1: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_3, arg5_1);  permute_3 = None
        convert_element_type_1: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        view_2: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(permute_1, _shape_param_2);  permute_1 = _shape_param_2 = None
        clone_1: "bf16[16, 512, 1024]" = torch.ops.aten.clone.default(view_2, memory_format = torch.contiguous_format);  view_2 = None
        view_3: "bf16[8192, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_3);  clone_1 = _shape_param_3 = None
        permute_4: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_3, [1, 0]);  view_3 = None
        mm_2: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_4, arg5_1);  permute_4 = None
        convert_element_type_2: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        clone_2: "bf16[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_4: "bf16[16, 512, 1024]" = torch.ops.aten.view.default(clone_2, _shape_param_4);  clone_2 = _shape_param_4 = None
        view_5: "bf16[8192, 1024]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        permute_5: "bf16[1024, 8192]" = torch.ops.aten.permute.default(view_5, [1, 0]);  view_5 = None
        mm_3: "bf16[1024, 1024]" = torch.ops.aten.mm.default(permute_5, arg5_1);  permute_5 = arg5_1 = None
        convert_element_type_3: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
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
