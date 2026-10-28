"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/MobileBertForMaskedLM/full_graph_000.py
Pattern hash: b73a9669de67
Shape hash: 9357cdfc
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
    def forward(self, arg0_1: "bf16[30522, 128]", arg1_1: "bf16[384, 30522]", arg2_1: "bf16[32768, 512]", arg3_1: "bf16[30522]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        permute: "bf16[128, 30522]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        cat: "bf16[512, 30522]" = torch.ops.aten.cat.default([permute, arg1_1]);  permute = arg1_1 = None
        constant_pad_nd: "bf16[512, 30528]" = torch.ops.aten.constant_pad_nd.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        mm: "bf16[32768, 30528]" = torch.ops.aten.mm.default(arg2_1, constant_pad_nd);  arg2_1 = constant_pad_nd = None
        slice_1: "bf16[32768, 30522]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -6);  mm = None
        view: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(slice_1, _shape_param_1);  slice_1 = _shape_param_1 = None
        add: "bf16[256, 128, 30522]" = torch.ops.aten.add.Tensor(view, arg3_1);  view = arg3_1 = None
        view_1: "bf16[32768, 30522]" = torch.ops.aten.view.default(add, _shape_param_2);  add = _shape_param_2 = None
        convert_element_type: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32)
        view_2: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        return (convert_element_type, view_2)



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
