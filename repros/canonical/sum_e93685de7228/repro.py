"""
Standalone repro captured via capture_hook.
Label: hf_DistillGPT2_train
Pattern hash: e93685de7228
Shape hash: 1a3ec418
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
    def forward(self, arg0_1: "bf16[32, 12, 512, 64]", arg1_1: "bf16[32, 12, 512, 64]", arg2_1: "bf16[32, 12, 512, 64]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4):
        # No stacktrace found for following nodes
        permute: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1, 3]);  arg0_1 = None
        view: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute, _shape_param_0);  permute = _shape_param_0 = None
        permute_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1, 3]);  arg1_1 = None
        view_1: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_1, _shape_param_1);  permute_1 = _shape_param_1 = None
        permute_2: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(arg2_1, [0, 2, 1, 3]);  arg2_1 = None
        view_2: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_2, _shape_param_2);  permute_2 = _shape_param_2 = None
        cat: "bf16[32, 512, 2304]" = torch.ops.aten.cat.default([view, view_2, view_1], 2);  view = view_2 = view_1 = None
        view_3: "bf16[16384, 2304]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        sum_1: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_3, [0], True, dtype = torch.float32)
        view_4: "f32[2304]" = torch.ops.aten.view.default(sum_1, _shape_param_4);  sum_1 = _shape_param_4 = None
        convert_element_type: "bf16[2304]" = torch.ops.prims.convert_element_type.default(view_4, torch.bfloat16);  view_4 = None
        convert_element_type_1: "f32[2304]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        return (view_3, convert_element_type_1)



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
