"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf-2-6-linux.aws.a100_graph23
Pattern hash: 28f1f945b54b
Shape hash: 08d5e95d
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
    def forward(self, convolution_18: "bf16[64, 512, 1, 1]", arg56_1: "bf16[512]", arg57_1: "bf16[512]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[64, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        view_default: "f32[64, 32, 16, 1]" = torch.ops.aten.view.default(convert_element_type_default, _shape_param_0);  convert_element_type_default = _shape_param_0 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(view_default, [2, 3], correction = 0, keepdim = True)
        getitem: "f32[64, 32, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[64, 32, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[64, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[64, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[64, 32, 16, 1]" = torch.ops.aten.sub.Tensor(view_default, getitem_1);  view_default = getitem_1 = None
        mul_tensor: "f32[64, 32, 16, 1]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        view_default_1: "f32[64, 512, 1, 1]" = torch.ops.aten.view.default(mul_tensor, _shape_param_1);  mul_tensor = _shape_param_1 = None
        unsqueeze_default: "bf16[1, 512]" = torch.ops.aten.unsqueeze.default(arg56_1, 0);  arg56_1 = None
        unsqueeze_default_1: "bf16[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "bf16[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_1: "f32[64, 512, 1, 1]" = torch.ops.aten.mul.Tensor(view_default_1, unsqueeze_default_2);  view_default_1 = unsqueeze_default_2 = None
        unsqueeze_default_3: "bf16[1, 512]" = torch.ops.aten.unsqueeze.default(arg57_1, 0);  arg57_1 = None
        unsqueeze_default_4: "bf16[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "bf16[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        add_tensor_1: "f32[64, 512, 1, 1]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        convert_element_type_default_1: "bf16[64, 512, 1, 1]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        relu_default: "bf16[64, 512, 1, 1]" = torch.ops.aten.relu.default(convert_element_type_default_1);  convert_element_type_default_1 = None
        return relu_default


def _default_make_inputs():
    return [
    torch.randn([64, 512, 1, 1], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512], dtype=torch.bfloat16, device='cuda'),
    [64, 32, 16, 1],  # _shape_param_0
    [64, 512, 1, 1],  # _shape_param_1
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
