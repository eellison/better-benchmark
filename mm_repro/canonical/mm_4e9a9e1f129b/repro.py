"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/GPTNeoForCausalLM/full_graph_000.py
Pattern hash: 4e9a9e1f129b
Shape hash: 7cb1f67d
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
    def forward(self, arg0_1: "bf16[2048, 2048]", arg1_1: "bf16[4096, 2048]", arg2_1: "bf16[2048, 2048]", arg3_1: "bf16[4096, 2048]", arg4_1: "bf16[2048, 2048]", arg5_1: "bf16[4096, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11):
        # No stacktrace found for following nodes
        permute: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[4096, 2048]" = torch.ops.aten.mm.default(arg1_1, permute);  arg1_1 = permute = None
        view: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        convert_element_type: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        expand: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        clone: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_2: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone, _shape_param_3);  clone = _shape_param_3 = None
        permute_2: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm_1: "bf16[4096, 2048]" = torch.ops.aten.mm.default(arg3_1, permute_2);  arg3_1 = permute_2 = None
        view_3: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_1, _shape_param_4);  mm_1 = _shape_param_4 = None
        view_4: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_3, _shape_param_5);  view_3 = _shape_param_5 = None
        permute_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        convert_element_type_1: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_3, torch.float32);  permute_3 = None
        permute_4: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_1, [0, 1, 3, 2]);  convert_element_type_1 = None
        expand_1: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_4, _shape_param_6);  permute_4 = _shape_param_6 = None
        clone_1: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_5: "f32[512, 128, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        permute_5: "bf16[2048, 2048]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        mm_2: "bf16[4096, 2048]" = torch.ops.aten.mm.default(arg5_1, permute_5);  arg5_1 = permute_5 = None
        view_6: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_2, _shape_param_8);  mm_2 = _shape_param_8 = None
        view_7: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_6, _shape_param_9);  view_6 = _shape_param_9 = None
        permute_6: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        expand_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(permute_6, _shape_param_10);  permute_6 = _shape_param_10 = None
        clone_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_8: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_2, _shape_param_11);  clone_2 = _shape_param_11 = None
        return (view_2, view_5, view_8)



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
