"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/XLNetLMHeadModel/full_graph_000.py
Pattern hash: 658428ef3758
Shape hash: f504d512
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
    def forward(self, arg0_1: "f32[1024, 16, 64]", arg1_1: "bf16[8192, 1024]", arg2_1: "f32[1024, 16, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        unsqueeze: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, 3);  convert_element_type = None
        unsqueeze_1: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 4);  unsqueeze = None
        view: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_0);  unsqueeze_1 = _shape_param_0 = None
        squeeze: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view, 0);  view = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg1_1, squeeze);  squeeze = None
        unsqueeze_2: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm, 0);  mm = None
        view_1: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_2, _shape_param_1);  unsqueeze_2 = _shape_param_1 = None
        permute: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_1, [0, 1, 3, 4, 2]);  view_1 = None
        view_2: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute, _shape_param_2);  permute = _shape_param_2 = None
        convert_element_type_1: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        unsqueeze_3: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 3);  convert_element_type_1 = None
        unsqueeze_4: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 4);  unsqueeze_3 = None
        view_3: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_4, _shape_param_3);  unsqueeze_4 = _shape_param_3 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_3, 0);  view_3 = None
        mm_1: "bf16[8192, 1024]" = torch.ops.aten.mm.default(arg1_1, squeeze_1);  arg1_1 = squeeze_1 = None
        unsqueeze_5: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_1, 0);  mm_1 = None
        view_4: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_5, _shape_param_4);  unsqueeze_5 = _shape_param_4 = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_4, [0, 1, 3, 4, 2]);  view_4 = None
        view_5: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_1, _shape_param_5);  permute_1 = _shape_param_5 = None
        unsqueeze_6: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_2, 4);  view_2 = None
        permute_2: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_6, [1, 2, 4, 0, 3]);  unsqueeze_6 = None
        permute_3: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_2, [0, 1, 4, 3, 2]);  permute_2 = None
        view_6: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_3, _shape_param_6);  permute_3 = _shape_param_6 = None
        unsqueeze_7: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_5, 4);  view_5 = None
        permute_4: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_7, [4, 1, 2, 3, 0]);  unsqueeze_7 = None
        permute_5: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_4, [1, 2, 4, 3, 0]);  permute_4 = None
        view_7: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_5, _shape_param_7);  permute_5 = _shape_param_7 = None
        permute_6: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_7, [0, 2, 1])
        permute_7: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1])
        return (view_6, view_7, permute_6, permute_7)



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
