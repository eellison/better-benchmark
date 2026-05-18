"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s6_g10
Pattern hash: 1a1af4176aa8
Shape hash: 059fe2f3
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_34: "bf16[4096, 480]", arg259_1: "bf16[240, 480]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        reshape_default: "bf16[256, 16, 480]" = torch.ops.aten.reshape.default(addmm_34, _shape_param_0);  addmm_34 = _shape_param_0 = None
        convert_element_type_default: "f32[256, 16, 480]" = torch.ops.prims.convert_element_type.default(reshape_default, torch.float32);  reshape_default = None
        neg_default: "f32[256, 16, 480]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[256, 16, 480]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[256, 16, 480]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[256, 16, 480]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "bf16[256, 16, 480]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        reshape_default_1: "bf16[4096, 480]" = torch.ops.aten.reshape.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        permute_default: "bf16[480, 240]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        return (reshape_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([4096, 480], dtype=torch.bfloat16, device='cuda'),
    torch.randn([240, 480], dtype=torch.bfloat16, device='cuda'),
    [256, 16, 480],  # _shape_param_0
    [4096, 480],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
