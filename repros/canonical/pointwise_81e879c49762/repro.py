"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-1-5-linux.aws.h100_graph33
Pattern hash: 81e879c49762
Shape hash: c4525f49
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
    def forward(self, addmm_66: "f16[8192, 768]", addmm_67: "f16[8192, 768]", arg189_1: "f32[768]", arg188_1: "f32[768, 768]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # No stacktrace found for following nodes
        view_default: "f16[16, 512, 768]" = torch.ops.aten.view.default(addmm_66, _shape_param_0);  addmm_66 = _shape_param_0 = None
        view_default_1: "f16[16, 512, 12, 64]" = torch.ops.aten.view.default(view_default, _shape_param_1);  view_default = _shape_param_1 = None
        permute_default: "f16[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2: "f16[16, 512, 768]" = torch.ops.aten.view.default(addmm_67, _shape_param_2);  addmm_67 = _shape_param_2 = None
        view_default_3: "f16[16, 512, 12, 64]" = torch.ops.aten.view.default(view_default_2, _shape_param_3);  view_default_2 = _shape_param_3 = None
        permute_default_1: "f16[16, 12, 512, 64]" = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        convert_element_type_default: "f16[768]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.float16);  arg189_1 = None
        convert_element_type_default_1: "f16[768, 768]" = torch.ops.prims.convert_element_type.default(arg188_1, torch.float16);  arg188_1 = None
        permute_default_2: "f16[768, 768]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        mul_scalar: "f16[16, 12, 512, 64]" = torch.ops.aten.mul.Scalar(permute_default, 0.3535533905932738);  permute_default = None
        permute_default_3: "f16[16, 12, 64, 512]" = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        mul_scalar_1: "f16[16, 12, 64, 512]" = torch.ops.aten.mul.Scalar(permute_default_3, 0.3535533905932738);  permute_default_3 = None
        expand_default: "f16[16, 12, 512, 64]" = torch.ops.aten.expand.default(mul_scalar, _shape_param_4);  mul_scalar = _shape_param_4 = None
        clone_default: "f16[16, 12, 512, 64]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_4: "f16[192, 512, 64]" = torch.ops.aten.view.default(clone_default, _shape_param_5);  clone_default = _shape_param_5 = None
        expand_default_1: "f16[16, 12, 64, 512]" = torch.ops.aten.expand.default(mul_scalar_1, _shape_param_6);  mul_scalar_1 = _shape_param_6 = None
        clone_default_1: "f16[16, 12, 64, 512]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_5: "f16[192, 64, 512]" = torch.ops.aten.view.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None
        return (convert_element_type_default, permute_default_2, view_default_4, view_default_5)


def _default_make_inputs():
    return [
    torch.randn([8192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float16, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [16, 512, 768],  # _shape_param_0
    [16, 512, -1, 64],  # _shape_param_1
    [16, 512, 768],  # _shape_param_2
    [16, 512, -1, 64],  # _shape_param_3
    [16, 12, 512, 64],  # _shape_param_4
    [192, 512, 64],  # _shape_param_5
    [16, 12, 64, 512],  # _shape_param_6
    [192, 64, 512],  # _shape_param_7
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
