"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-4-5-linux.aws.h100_graph40
Pattern hash: 9d6fe3b8f034
Shape hash: 5807db38
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
    def forward(self, addmm: "f16[128, 1024]", addmm_1: "f16[128, 1024]", arg8_1: "f32[1024]", arg7_1: "f32[1024, 1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # No stacktrace found for following nodes
        view_default: "f16[1, 128, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        mul_tensor: "f16[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_default, 0.125);  view_default = None
        view_default_1: "f16[1, 128, 1024]" = torch.ops.aten.view.default(addmm_1, _shape_param_1);  addmm_1 = _shape_param_1 = None
        convert_element_type_default: "f16[1024]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float16);  arg8_1 = None
        convert_element_type_default_1: "f16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float16);  arg7_1 = None
        permute_default: "f16[1024, 1024]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        view_default_2: "f16[1, 128, 16, 64]" = torch.ops.aten.view.default(view_default_1, _shape_param_2);  view_default_1 = _shape_param_2 = None
        permute_default_1: "f16[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        view_default_3: "f16[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_tensor, _shape_param_3);  mul_tensor = _shape_param_3 = None
        permute_default_2: "f16[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        view_default_4: "f16[16, 128, 64]" = torch.ops.aten.view.default(permute_default_2, _shape_param_4);  permute_default_2 = _shape_param_4 = None
        view_default_5: "f16[16, 128, 64]" = torch.ops.aten.view.default(permute_default_1, _shape_param_5);  permute_default_1 = _shape_param_5 = None
        permute_default_3: "f16[16, 64, 128]" = torch.ops.aten.permute.default(view_default_5, [0, 2, 1]);  view_default_5 = None
        return (convert_element_type_default, permute_default, view_default_4, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([128, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([128, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [1, 128, 1024],  # _shape_param_0
    [1, 128, 1024],  # _shape_param_1
    [1, 128, -1, 64],  # _shape_param_2
    [1, 128, 16, 64],  # _shape_param_3
    [16, 128, 64],  # _shape_param_4
    [16, 128, 64],  # _shape_param_5
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
