"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-4-5-linux.aws.h100_graph40
Pattern hash: 83bb69bef762
Shape hash: 7226088c
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
    def forward(self, addmm_3: "f16[128, 1024]", arg2_1: "f32[1, 128, 1024]", arg12_1: "f32[1024]", arg13_1: "f32[1024]", arg15_1: "f32[4096]", arg14_1: "f32[4096, 1024]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f16[1, 128, 1024]" = torch.ops.aten.view.default(addmm_3, _shape_param_0);  addmm_3 = _shape_param_0 = None
        native_dropout_default = torch.ops.aten.native_dropout.default(view_default, 0.1, True);  view_default = None
        getitem: "f16[1, 128, 1024]" = native_dropout_default[0]
        getitem_1: "b8[1, 128, 1024]" = native_dropout_default[1];  native_dropout_default = None
        add_tensor: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(arg2_1, getitem);  arg2_1 = getitem = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem_2: "f32[1, 128, 1]" = var_mean_correction[0]
        getitem_3: "f32[1, 128, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_3);  add_tensor = getitem_3 = None
        mul_tensor: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, arg12_1);  mul_tensor = arg12_1 = None
        add_tensor_2: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg13_1);  mul_tensor_1 = arg13_1 = None
        convert_element_type_default: "f16[4096]" = torch.ops.prims.convert_element_type.default(arg15_1, torch.float16);  arg15_1 = None
        convert_element_type_default_1: "f16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.float16);  arg14_1 = None
        convert_element_type_default_2: "f16[1, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float16);  add_tensor_2 = None
        view_default_1: "f16[128, 1024]" = torch.ops.aten.view.default(convert_element_type_default_2, _shape_param_1);  convert_element_type_default_2 = _shape_param_1 = None
        permute_default: "f16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        return (convert_element_type_default, view_default_1, permute_default, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([128, 1024], dtype=torch.float16, device='cuda'),
    torch.randn([1, 128, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([4096], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 1024], dtype=torch.float32, device='cuda'),
    [1, 128, 1024],  # _shape_param_0
    [128, 1024],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
