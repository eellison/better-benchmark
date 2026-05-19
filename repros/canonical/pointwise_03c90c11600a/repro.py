"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-1-5-linux.aws.a100_graph2
Pattern hash: 03c90c11600a
Shape hash: 454672fd
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
    def forward(self, addmm_70: "bf16[1024, 3072]", arg191_1: "bf16[768, 3072]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "bf16[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_70, _shape_param_0);  addmm_70 = _shape_param_0 = None
        convert_element_type_default: "f32[1, 1024, 3072]" = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        mul_tensor: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5)
        mul_tensor_1: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.7071067811865476);  convert_element_type_default = None
        erf_default: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1: "bf16[1, 1024, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        view_default_1: "bf16[1024, 3072]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        permute_default: "bf16[3072, 768]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        return (view_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1024, 3072], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768, 3072], dtype=torch.bfloat16, device='cuda'),
    [1, 1024, 3072],  # _shape_param_0
    [1024, 3072],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
