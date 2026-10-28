"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/LayoutLMForMaskedLM/full_graph_000.py
Pattern hash: 24f1e7660a5f
Shape hash: decbd795
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
    def forward(self, arg0_1: "f32[768]", arg1_1: "f32[768, 768]", arg2_1: "bf16[16384, 768]", arg3_1: "f32[768]", arg4_1: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        permute: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type, arg2_1, permute);  convert_element_type = permute = None
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        view_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        convert_element_type_2: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_3: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        permute_2: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_3, [1, 0]);  convert_element_type_3 = None
        addmm_1: "bf16[16384, 768]" = torch.ops.aten.addmm.default(convert_element_type_2, arg2_1, permute_2);  convert_element_type_2 = arg2_1 = permute_2 = None
        view_2: "bf16[32, 512, 768]" = torch.ops.aten.view.default(addmm_1, _shape_param_2);  addmm_1 = _shape_param_2 = None
        view_3: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        permute_3: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        permute_4: "bf16[32, 12, 64, 512]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
        expand: "bf16[32, 12, 512, 64]" = torch.ops.aten.expand.default(permute_1, _shape_param_4);  permute_1 = _shape_param_4 = None
        clone: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_4: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_5);  clone = _shape_param_5 = None
        expand_1: "bf16[32, 12, 64, 512]" = torch.ops.aten.expand.default(permute_4, _shape_param_6);  permute_4 = _shape_param_6 = None
        clone_1: "bf16[32, 12, 64, 512]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_5: "bf16[384, 64, 512]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        permute_5: "bf16[384, 64, 512]" = torch.ops.aten.permute.default(view_4, [0, 2, 1])
        permute_6: "bf16[384, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1])
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
