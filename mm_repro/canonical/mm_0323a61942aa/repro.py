"""
Standalone repro captured via capture_hook.
Label: repros/models/timm/train/mobilevit_s/full_graph_001.py
Pattern hash: 0323a61942aa
Shape hash: 8da39bde
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
    def forward(self, arg0_1: "bf16[512, 4, 16, 64]", arg1_1: "bf16[512, 4, 16, 64]", arg2_1: "bf16[512, 4, 16, 64]", arg3_1: "bf16[8192, 240]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        constant_pad_nd: "bf16[512, 4, 16, 60]" = torch.ops.aten.constant_pad_nd.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        constant_pad_nd_1: "bf16[512, 4, 16, 60]" = torch.ops.aten.constant_pad_nd.default(arg1_1, _shape_param_1);  arg1_1 = _shape_param_1 = None
        constant_pad_nd_2: "bf16[512, 4, 16, 60]" = torch.ops.aten.constant_pad_nd.default(arg2_1, _shape_param_2);  arg2_1 = _shape_param_2 = None
        cat: "bf16[1536, 4, 16, 60]" = torch.ops.aten.cat.default([constant_pad_nd_2, constant_pad_nd_1, constant_pad_nd]);  constant_pad_nd_2 = constant_pad_nd_1 = constant_pad_nd = None
        view: "bf16[3, 512, 4, 16, 60]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        permute: "bf16[512, 16, 3, 4, 60]" = torch.ops.aten.permute.default(view, [1, 3, 0, 2, 4]);  view = None
        clone: "bf16[512, 16, 3, 4, 60]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[512, 16, 720]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        view_2: "bf16[8192, 720]" = torch.ops.aten.view.default(view_1, _shape_param_5);  view_1 = _shape_param_5 = None
        permute_1: "bf16[720, 8192]" = torch.ops.aten.permute.default(view_2, [1, 0]);  view_2 = None
        mm: "bf16[720, 240]" = torch.ops.aten.mm.default(permute_1, arg3_1);  permute_1 = arg3_1 = None
        convert_element_type: "f32[720, 240]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type



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
