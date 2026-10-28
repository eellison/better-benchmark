"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/AllenaiLongformerBase/full_graph_002.py
Pattern hash: 32e532f427ac
Shape hash: 5be248ab
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
    def forward(self, arg0_1: "f32[8, 1024, 768]", arg1_1: "f32[768]", arg2_1: "f32[768, 768]", arg3_1: "f32[768]", arg4_1: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14):
        # No stacktrace found for following nodes
        permute: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0, 2]);  arg0_1 = None
        convert_element_type: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_1: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        convert_element_type_2: "bf16[1024, 8, 768]" = torch.ops.prims.convert_element_type.default(permute, torch.bfloat16);  permute = None
        permute_1: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        clone: "bf16[1024, 8, 768]" = torch.ops.aten.clone.default(convert_element_type_2, memory_format = torch.contiguous_format);  convert_element_type_2 = None
        view: "bf16[8192, 768]" = torch.ops.aten.view.default(clone, _shape_param_0);  clone = _shape_param_0 = None
        mm: "bf16[8192, 768]" = torch.ops.aten.mm.default(view, permute_1);  permute_1 = None
        view_1: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm, _shape_param_1);  mm = _shape_param_1 = None
        add: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_1, convert_element_type);  view_1 = convert_element_type = None
        convert_element_type_3: "bf16[768]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_4: "bf16[768, 768]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        permute_2: "bf16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        mm_1: "bf16[8192, 768]" = torch.ops.aten.mm.default(view, permute_2);  view = permute_2 = None
        view_2: "bf16[1024, 8, 768]" = torch.ops.aten.view.default(mm_1, _shape_param_2);  mm_1 = _shape_param_2 = None
        add_1: "bf16[1024, 8, 768]" = torch.ops.aten.add.Tensor(view_2, convert_element_type_3);  view_2 = convert_element_type_3 = None
        div: "bf16[1024, 8, 768]" = torch.ops.aten.div.Tensor(add, 8.0);  add = None
        view_3: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(add_1, _shape_param_3);  add_1 = _shape_param_3 = None
        permute_3: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_3, [1, 0, 2, 3]);  view_3 = None
        permute_4: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_3, [0, 2, 1, 3]);  permute_3 = None
        view_4: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_4, _shape_param_4);  permute_4 = _shape_param_4 = None
        view_5: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_4, _shape_param_5);  view_4 = _shape_param_5 = None
        as_strided: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_5, _shape_param_6, _shape_param_7);  view_5 = _shape_param_6 = _shape_param_7 = None
        unsqueeze: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided, 4);  as_strided = None
        permute_5: "bf16[96, 3, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze, [0, 1, 4, 2, 3]);  unsqueeze = None
        view_6: "bf16[1024, 8, 12, 64]" = torch.ops.aten.view.default(div, _shape_param_8);  div = _shape_param_8 = None
        permute_6: "bf16[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_6, [1, 0, 2, 3]);  view_6 = None
        permute_7: "bf16[8, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1, 3]);  permute_6 = None
        view_7: "bf16[96, 1024, 64]" = torch.ops.aten.view.default(permute_7, _shape_param_9);  permute_7 = _shape_param_9 = None
        view_8: "bf16[96, 2, 512, 64]" = torch.ops.aten.view.default(view_7, _shape_param_10);  view_7 = _shape_param_10 = None
        as_strided_1: "bf16[96, 3, 512, 64]" = torch.ops.aten.as_strided.default(view_8, _shape_param_11, _shape_param_12);  view_8 = _shape_param_11 = _shape_param_12 = None
        unsqueeze_1: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.unsqueeze.default(as_strided_1, 4);  as_strided_1 = None
        clone_1: "bf16[96, 3, 512, 64, 1]" = torch.ops.aten.clone.default(unsqueeze_1, memory_format = torch.contiguous_format);  unsqueeze_1 = None
        view_9: "bf16[288, 512, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_13);  clone_1 = _shape_param_13 = None
        permute_8: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.permute.default(permute_5, [0, 1, 4, 3, 2]);  permute_5 = None
        clone_2: "bf16[96, 3, 64, 512, 1]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_10: "bf16[288, 64, 512]" = torch.ops.aten.view.default(clone_2, _shape_param_14);  clone_2 = _shape_param_14 = None
        permute_9: "bf16[288, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1])
        permute_10: "bf16[288, 512, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1])
        return (view_9, view_10, permute_9, permute_10)



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
