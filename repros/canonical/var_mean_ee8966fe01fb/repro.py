"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf_cuda_h100-2-5-linux.aws.h100_graph36
Pattern hash: ee8966fe01fb
Shape hash: 38e587ff
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
    def forward(self, addmm_21: "bf16[512, 768]", add_43: "bf16[1, 512, 768]", arg69_1: "bf16[768]", arg70_1: "bf16[768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "bf16[1, 512, 768]" = torch.ops.aten.view.default(addmm_21, _shape_param_0);  addmm_21 = _shape_param_0 = None
        add_tensor: "bf16[1, 512, 768]" = torch.ops.aten.add.Tensor(view_default, add_43);  view_default = add_43 = None
        convert_element_type_default: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg69_1);  mul_tensor = arg69_1 = None
        add_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg70_1);  mul_tensor_1 = arg70_1 = None
        convert_element_type_default_1: "bf16[1, 512, 768]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default_1: "bf16[512, 768]" = torch.ops.aten.view.default(convert_element_type_default_1, _shape_param_1);  convert_element_type_default_1 = _shape_param_1 = None
        return view_default_1


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768], dtype=torch.bfloat16, device='cuda'),
    torch.randn([768], dtype=torch.bfloat16, device='cuda'),
    [1, 512, 768],  # _shape_param_0
    [-1, 768],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
