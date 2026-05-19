"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-2-7-linux.aws.h100_graph49
Pattern hash: 0729c8502c7f
Shape hash: fa03e7ce
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
    def forward(self, addmm_47: "f16[1584, 768]", add_80: "f32[8, 198, 768]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg153_1: "f32[1000]", arg152_1: "f32[1000, 768]", arg155_1: "f32[1000]", arg154_1: "f32[1000, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f16[8, 198, 768]" = torch.ops.aten.view.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None
        add_tensor: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_80, view_default);  add_80 = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 198, 1]" = var_mean_correction[0]
        getitem_1: "f32[8, 198, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt_default: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_tensor, getitem_1);  add_tensor = getitem_1 = None
        mul_tensor: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg150_1);  mul_tensor = arg150_1 = None
        add_tensor_2: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg151_1);  mul_tensor_1 = arg151_1 = None
        select_int: "f32[8, 768]" = torch.ops.aten.select.int(add_tensor_2, 1, 0)
        select_int_1: "f32[8, 768]" = torch.ops.aten.select.int(add_tensor_2, 1, 1);  add_tensor_2 = None
        convert_element_type_default: "f16[1000]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.float16);  arg153_1 = None
        convert_element_type_default_1: "f16[1000, 768]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.float16);  arg152_1 = None
        convert_element_type_default_2: "f16[8, 768]" = torch.ops.prims.convert_element_type.default(select_int, torch.float16);  select_int = None
        permute_default: "f16[768, 1000]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        convert_element_type_default_3: "f16[1000]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.float16);  arg155_1 = None
        convert_element_type_default_4: "f16[1000, 768]" = torch.ops.prims.convert_element_type.default(arg154_1, torch.float16);  arg154_1 = None
        convert_element_type_default_5: "f16[8, 768]" = torch.ops.prims.convert_element_type.default(select_int_1, torch.float16);  select_int_1 = None
        permute_default_1: "f16[768, 1000]" = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        return (convert_element_type_default, convert_element_type_default_2, permute_default, convert_element_type_default_3, convert_element_type_default_5, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([1584, 768], dtype=torch.float16, device='cuda'),
    torch.randn([8, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    [8, 198, 768],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
