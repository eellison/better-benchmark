"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-5-5-linux.aws.a100_graph11
Pattern hash: 1436270a80ef
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
    def forward(self, addmm_45: "bf16[512, 1024]", convert_element_type_182: "bf16[512, 1, 1024]", _param_constant183: "bf16[1024]", _param_constant184: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2):
        # No stacktrace found for following nodes
        view_default: "bf16[512, 1, 1024]" = torch.ops.aten.view.default(addmm_45, _shape_param_0);  addmm_45 = _shape_param_0 = None
        add_tensor: "bf16[512, 1, 1024]" = torch.ops.aten.add.Tensor(view_default, convert_element_type_182);  view_default = convert_element_type_182 = None
        convert_element_type_default: "f32[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[512, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1: "f32[512, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[512, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor: "f32[512, 1, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1);  convert_element_type_default = getitem_1 = None
        mul_tensor: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[512, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_tensor, _param_constant183);  mul_tensor = _param_constant183 = None
        add_tensor_2: "f32[512, 1, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_1, _param_constant184);  mul_tensor_1 = _param_constant184 = None
        convert_element_type_default_1: "bf16[512, 1, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        unsqueeze_default: "bf16[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 3)
        unsqueeze_default_1: "bf16[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        permute_default: "bf16[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_default_1, [0, 1, 3, 4, 2]);  unsqueeze_default_1 = None
        permute_default_1: "bf16[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_default, [0, 4, 1, 2, 3]);  permute_default = None
        view_default_1: "bf16[1, 512, 1024]" = torch.ops.aten.view.default(permute_default_1, _shape_param_1);  permute_default_1 = _shape_param_1 = None
        squeeze_dim: "bf16[512, 1024]" = torch.ops.aten.squeeze.dim(view_default_1, 0);  view_default_1 = None
        unsqueeze_default_2: "bf16[512, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 3);  convert_element_type_default_1 = None
        unsqueeze_default_3: "bf16[512, 1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        permute_default_2: "bf16[512, 1, 1, 1, 1024]" = torch.ops.aten.permute.default(unsqueeze_default_3, [0, 1, 3, 4, 2]);  unsqueeze_default_3 = None
        permute_default_3: "bf16[512, 1024, 1, 1, 1]" = torch.ops.aten.permute.default(permute_default_2, [0, 4, 1, 2, 3]);  permute_default_2 = None
        view_default_2: "bf16[1, 512, 1024]" = torch.ops.aten.view.default(permute_default_3, _shape_param_2);  permute_default_3 = _shape_param_2 = None
        squeeze_dim_1: "bf16[512, 1024]" = torch.ops.aten.squeeze.dim(view_default_2, 0);  view_default_2 = None
        return (squeeze_dim, squeeze_dim_1)


def _default_make_inputs():
    return [
    torch.randn([512, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512, 1, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024], dtype=torch.bfloat16, device='cuda'),
    [512, 1, 1024],  # _shape_param_0
    [1, 512, 1024],  # _shape_param_1
    [1, 512, 1024],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
