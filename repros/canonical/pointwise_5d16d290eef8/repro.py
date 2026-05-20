"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-6-7-linux.aws.h100_graph58
Pattern hash: 5d16d290eef8
Shape hash: ef7c876b
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([512, 480], bf16), T([240, 480], bf16), S([32, 16, 480]), S([512, 480]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_34: "bf16[512, 480]", arg259_1: "bf16[240, 480]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "bf16[32, 16, 480]" = torch.ops.aten.view.default(addmm_34, _shape_param_0);  addmm_34 = _shape_param_0 = None
        convert_element_type_default: "f32[32, 16, 480]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        neg_default: "f32[32, 16, 480]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[32, 16, 480]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[32, 16, 480]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[32, 16, 480]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "bf16[32, 16, 480]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        view_default_1: "bf16[512, 480]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        permute_default: "bf16[480, 240]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        return (view_default_1, permute_default)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
