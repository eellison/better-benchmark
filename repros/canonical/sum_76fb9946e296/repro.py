"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 76fb9946e296
Shape hash: 07e248d7
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
    def forward(self, arg0_1: "bf16[512, 128, 80]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3):
        # No stacktrace found for following nodes
        view: "bf16[16, 32, 128, 80]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "bf16[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(view, 0.334370152488211);  view = None
        permute: "bf16[16, 128, 32, 80]" = torch.ops.aten.permute.default(mul, [0, 2, 1, 3]);  mul = None
        clone: "bf16[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        view_1: "bf16[16, 128, 2560]" = torch.ops.aten.view.default(clone, _shape_param_1);  clone = _shape_param_1 = None
        view_2: "bf16[2048, 2560]" = torch.ops.aten.view.default(view_1, _shape_param_2);  view_1 = _shape_param_2 = None
        permute_1: "bf16[2560, 2048]" = torch.ops.aten.permute.default(view_2, [1, 0])
        sum_1: "f32[1, 2560]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32)
        view_3: "f32[2560]" = torch.ops.aten.view.default(sum_1, _shape_param_3);  sum_1 = _shape_param_3 = None
        convert_element_type: "bf16[2560]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_1: "f32[2560]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        return (view_2, permute_1, convert_element_type_1)



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
