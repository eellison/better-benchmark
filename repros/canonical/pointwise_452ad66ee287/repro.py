"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train
Pattern hash: 452ad66ee287
Shape hash: 1398f333
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
    def forward(self, arg0_1: "bf16[32768, 128]", arg1_1: "f32[128]", arg2_1: "f32[128]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view: "bf16[256, 128, 128]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        mul: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view, arg1_1);  view = arg1_1 = None
        add: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul, arg2_1);  mul = arg2_1 = None
        convert_element_type: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        view_1: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type, _shape_param_1);  convert_element_type = _shape_param_1 = None
        return view_1



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
