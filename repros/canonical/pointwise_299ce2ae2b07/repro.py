"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_train
Pattern hash: 299ce2ae2b07
Shape hash: 01e90f6e
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
    def forward(self, arg0_1: "i64[64, 128]", arg1_1: "i32[64, 128]", arg2_1: "f32[1026, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "i32[64, 128]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.int32);  arg0_1 = None
        add: "i32[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type, 0);  convert_element_type = None
        mul: "i32[64, 128]" = torch.ops.aten.mul.Tensor(add, arg1_1);  add = arg1_1 = None
        convert_element_type_1: "i64[64, 128]" = torch.ops.prims.convert_element_type.default(mul, torch.int64);  mul = None
        add_1: "i64[64, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1);  convert_element_type_1 = None
        view: "i64[8192]" = torch.ops.aten.view.default(add_1, [-1]);  add_1 = None
        index: "f32[8192, 1024]" = torch.ops.aten.index.Tensor(arg2_1, [view]);  arg2_1 = view = None
        view_1: "f32[64, 128, 1024]" = torch.ops.aten.view.default(index, _shape_param_0);  index = _shape_param_0 = None
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
