"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train
Pattern hash: ee05f315ed84
Shape hash: 18d1aea1
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
    def forward(self, arg0_1: "bf16[8192, 1024]", arg1_1: "f32[8, 1024, 1024]", arg2_1: "bf16[8192, 1024]", arg3_1: "bf16[8192, 1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view: "bf16[8, 1024, 1024]" = torch.ops.aten.view.default(arg0_1, _shape_param_0);  arg0_1 = _shape_param_0 = None
        convert_element_type: "f32[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        add: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(arg1_1, convert_element_type);  arg1_1 = convert_element_type = None
        view_1: "bf16[8, 1024, 1024]" = torch.ops.aten.view.default(arg2_1, _shape_param_1);  arg2_1 = _shape_param_1 = None
        convert_element_type_1: "f32[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        add_1: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add, convert_element_type_1);  add = convert_element_type_1 = None
        view_2: "bf16[8, 1024, 1024]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        add_2: "f32[8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_1, convert_element_type_2);  add_1 = convert_element_type_2 = None
        return add_2



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
