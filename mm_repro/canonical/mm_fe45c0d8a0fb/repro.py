"""
Standalone repro captured via capture_hook.
Label: repros/models/hf/train/BlenderbotForCausalLM/full_graph_007.py
Pattern hash: fe45c0d8a0fb
Shape hash: b80dfbfb
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
    def forward(self, arg0_1: "f32[32, 128, 2560]", arg1_1: "b8[32, 128, 2560]", arg2_1: "bf16[4096, 10240]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_1: "bf16[32, 128, 2560]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        mul: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_1: "bf16[32, 128, 2560]" = torch.ops.aten.mul.Tensor(convert_element_type, mul);  convert_element_type = mul = None
        view: "bf16[4096, 2560]" = torch.ops.aten.view.default(mul_1, _shape_param_0);  mul_1 = _shape_param_0 = None
        permute: "bf16[2560, 4096]" = torch.ops.aten.permute.default(view, [1, 0]);  view = None
        mm: "bf16[2560, 10240]" = torch.ops.aten.mm.default(permute, arg2_1);  permute = arg2_1 = None
        convert_element_type_2: "f32[2560, 10240]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        return convert_element_type_2



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
