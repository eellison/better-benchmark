"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/google/gemma-3-4b-it/full_graph_000.py
Pattern hash: c926c18e5e99
Shape hash: 0e79adbe
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
    def forward(self, arg0_1: "bf16[2048, 2560]", arg1_1: "bf16[1000, 2560]", arg2_1: "bf16[1024, 2560]", arg3_1: "bf16[1000, 2560]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        permute: "bf16[2560, 2048]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[1000, 2048]" = torch.ops.aten.mm.default(arg1_1, permute);  arg1_1 = permute = None
        view: "bf16[1, 1000, 2048]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        view_1: "bf16[1, 1000, 8, 256]" = torch.ops.aten.view.default(view, _shape_param_1);  view = _shape_param_1 = None
        permute_1: "bf16[1, 8, 1000, 256]" = torch.ops.aten.permute.default(view_1, [0, 2, 1, 3]);  view_1 = None
        convert_element_type: "f32[1, 8, 1000, 256]" = torch.ops.prims.convert_element_type.default(permute_1, torch.float32);  permute_1 = None
        pow_1: "f32[1, 8, 1000, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2);  convert_element_type = None
        permute_2: "bf16[2560, 1024]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm_1: "bf16[1000, 1024]" = torch.ops.aten.mm.default(arg3_1, permute_2);  arg3_1 = permute_2 = None
        view_2: "bf16[1, 1000, 1024]" = torch.ops.aten.view.default(mm_1, _shape_param_2);  mm_1 = _shape_param_2 = None
        view_3: "bf16[1, 1000, 4, 256]" = torch.ops.aten.view.default(view_2, _shape_param_3);  view_2 = _shape_param_3 = None
        permute_3: "bf16[1, 4, 1000, 256]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        convert_element_type_1: "f32[1, 4, 1000, 256]" = torch.ops.prims.convert_element_type.default(permute_3, torch.float32);  permute_3 = None
        pow_2: "f32[1, 4, 1000, 256]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1, 2);  convert_element_type_1 = None
        return (pow_1, pow_2)



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
