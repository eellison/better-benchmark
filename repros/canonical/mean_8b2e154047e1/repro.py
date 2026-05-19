"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf-4-6-linux.aws.a100_graph17
Pattern hash: 8b2e154047e1
Shape hash: 97698c9c
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convolution_80: "bf16[128, 2304, 9, 9]", arg220_1: "bf16[1000, 2304]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[128, 2304, 9, 9]" = torch.ops.prims.convert_element_type.default(convolution_80, torch.float32);  convolution_80 = None
        neg_default: "f32[128, 2304, 9, 9]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[128, 2304, 9, 9]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 2304, 9, 9]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 2304, 9, 9]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor);  convert_element_type_default = add_tensor = None
        convert_element_type_default_1: "bf16[128, 2304, 9, 9]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        mean_dim: "bf16[128, 2304, 1, 1]" = torch.ops.aten.mean.dim(convert_element_type_default_1, [-1, -2], True);  convert_element_type_default_1 = None
        view_default: "bf16[128, 2304]" = torch.ops.aten.view.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        permute_default: "bf16[2304, 1000]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
        return (view_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([128, 2304, 9, 9], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1000, 2304], dtype=torch.bfloat16, device='cuda'),
    [128, 2304],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
