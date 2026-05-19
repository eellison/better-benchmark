"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-4-5-linux.aws.h100_graph40
Pattern hash: ecdcda9deb72
Shape hash: 351c30ff
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
    def forward(self, arg2_1: "f32[1, 128, 1024]", arg0_1: "f32[1024]", arg1_1: "f32[1024]", arg4_1: "f32[1024]", arg3_1: "f32[1024, 1024]", arg6_1: "f32[1024]", arg5_1: "f32[1024, 1024]", _shape_param_0):
        # No stacktrace found for following nodes
        var_mean_correction = torch.ops.aten.var_mean.correction(arg2_1, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(arg2_1, getitem_1);  arg2_1 = getitem_1 = None
        mul_tensor: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg0_1);  mul_tensor = arg0_1 = None
        add_tensor_1: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg1_1);  mul_tensor_1 = arg1_1 = None
        convert_element_type_default: "f16[1024]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float16);  arg4_1 = None
        convert_element_type_default_1: "f16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float16);  arg3_1 = None
        convert_element_type_default_2: "f16[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float16);  add_tensor_1 = None
        view_default: "f16[128, 1024]" = torch.ops.aten.view.default(convert_element_type_default_2, _shape_param_0);  convert_element_type_default_2 = _shape_param_0 = None
        permute_default: "f16[1024, 1024]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        convert_element_type_default_3: "f16[1024]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.float16);  arg6_1 = None
        convert_element_type_default_4: "f16[1024, 1024]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.float16);  arg5_1 = None
        permute_default_1: "f16[1024, 1024]" = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        return (convert_element_type_default, view_default, permute_default, convert_element_type_default_3, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([1, 128, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 1024], dtype=torch.float32, device='cuda'),
    [128, 1024],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
