"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/GPTNeoForCausalLM/full_graph_000.py
Pattern hash: 99f07abee04d
Shape hash: f81440f0
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
    def forward(self, arg0_1: "f32[2048, 2048]", arg1_1: "bf16[4096, 2048]", arg2_1: "f32[2048, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        permute: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type, [1, 0]);  convert_element_type = None
        mm: "bf16[4096, 2048]" = torch.ops.aten.mm.default(arg1_1, permute);  permute = None
        view: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type_1: "bf16[2048, 2048]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        permute_1: "bf16[2048, 2048]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        mm_1: "bf16[4096, 2048]" = torch.ops.aten.mm.default(arg1_1, permute_1);  arg1_1 = permute_1 = None
        view_1: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_1, _shape_param_1);  mm_1 = _shape_param_1 = None
        view_2: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view, _shape_param_2);  view = _shape_param_2 = None
        permute_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        view_3: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        permute_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        convert_element_type_2: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_3, torch.float32);  permute_3 = None
        permute_4: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_2, [0, 1, 3, 2]);  convert_element_type_2 = None
        convert_element_type_3: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_4, torch.bfloat16);  permute_4 = None
        convert_element_type_4: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_2, torch.bfloat16);  permute_2 = None
        expand: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_4, _shape_param_4);  convert_element_type_4 = _shape_param_4 = None
        clone: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_4: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        expand_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type_3, _shape_param_6);  convert_element_type_3 = _shape_param_6 = None
        clone_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_5: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        permute_5: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_4, [0, 2, 1])
        permute_6: "bf16[512, 128, 128]" = torch.ops.aten.permute.default(view_5, [0, 2, 1])
        return (view_4, view_5, permute_5, permute_6)



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
