"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-5-5-linux.aws.a100_graph11
Pattern hash: 19e492b6d6ab
Shape hash: 9c8d47f2
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
    def forward(self, addmm_47: "bf16[512, 1024]", convert_element_type_190: "bf16[512, 1, 1024]", _param_constant191: "bf16[1024]", _param_constant192: "bf16[1024]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "bf16[512, 1, 1024]" = torch.ops.aten.view.default(addmm_47, _shape_param_0);  addmm_47 = _shape_param_0 = None
        add_tensor: "bf16[512, 1, 1024]" = torch.ops.aten.add.Tensor(view_default, convert_element_type_190);  view_default = convert_element_type_190 = None
        convert_element_type_default: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        mul_tensor: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, _param_constant191);  mul_tensor = _param_constant191 = None
        add_tensor_2: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, _param_constant192);  mul_tensor_1 = _param_constant192 = None
        convert_element_type_default_1: "bf16[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        permute_default: "bf16[1, 512, 1024]" = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0, 2]);  convert_element_type_default_1 = None
        view_default_1: "bf16[512, 1024]" = torch.ops.aten.view.default(permute_default, _shape_param_1);  permute_default = _shape_param_1 = None
        return view_default_1


def _default_make_inputs():
    return [
    torch.randn([512, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024], dtype=torch.bfloat16, device='cuda'),
    [512, 1, 1024],  # _shape_param_0
    [512, 1024],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
