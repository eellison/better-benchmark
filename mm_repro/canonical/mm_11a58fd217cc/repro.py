"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/infer/google/gemma-2-2b/full_graph_000.py
Pattern hash: 11a58fd217cc
Shape hash: 8817e06c
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
    def forward(self, arg0_1: "bf16[9216, 2304]", arg1_1: "bf16[1000, 2304]", arg2_1: "bf16[9216, 2304]", arg3_1: "bf16[1000, 2304]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        permute: "bf16[2304, 9216]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[1000, 9216]" = torch.ops.aten.mm.default(arg1_1, permute);  arg1_1 = permute = None
        view: "bf16[1, 1000, 9216]" = torch.ops.aten.view.default(mm, _shape_param_0);  mm = _shape_param_0 = None
        convert_element_type: "f32[1, 1000, 9216]" = torch.ops.prims.convert_element_type.default(view, torch.float32);  view = None
        mul: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type, 0.5)
        mul_1: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type, convert_element_type)
        mul_2: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(mul_1, convert_element_type);  mul_1 = None
        mul_3: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(mul_2, 0.044715);  mul_2 = None
        add: "f32[1, 1000, 9216]" = torch.ops.aten.add.Tensor(convert_element_type, mul_3);  convert_element_type = mul_3 = None
        mul_4: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(add, 0.7978845608028654);  add = None
        tanh: "f32[1, 1000, 9216]" = torch.ops.aten.tanh.default(mul_4);  mul_4 = None
        add_1: "f32[1, 1000, 9216]" = torch.ops.aten.add.Tensor(tanh, 1);  tanh = None
        mul_5: "f32[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        convert_element_type_1: "bf16[1, 1000, 9216]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        permute_1: "bf16[2304, 9216]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm_1: "bf16[1000, 9216]" = torch.ops.aten.mm.default(arg3_1, permute_1);  arg3_1 = permute_1 = None
        view_1: "bf16[1, 1000, 9216]" = torch.ops.aten.view.default(mm_1, _shape_param_1);  mm_1 = _shape_param_1 = None
        mul_6: "bf16[1, 1000, 9216]" = torch.ops.aten.mul.Tensor(convert_element_type_1, view_1);  convert_element_type_1 = view_1 = None
        view_2: "bf16[1000, 9216]" = torch.ops.aten.view.default(mul_6, _shape_param_2);  mul_6 = _shape_param_2 = None
        return view_2



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
