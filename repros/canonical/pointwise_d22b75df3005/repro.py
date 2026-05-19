"""
Standalone repro captured via capture_hook.
Label: inductor_torchbench_perf_cuda_h100-2-9-linux.aws.h100_graph63
Pattern hash: d22b75df3005
Shape hash: 38f1d9ac
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
    def forward(self, arg13_1: "bf16[512]", arg14_1: "bf16[512]", convolution_3: "bf16[256, 512, 4, 4]", arg15_1: "bf16[512]", arg16_1: "bf16[512]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[512]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        convert_element_type_default_1: "f32[512]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.float32);  arg14_1 = None
        add_tensor: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[512]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[256, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_default_1);  convolution_3 = unsqueeze_default_1 = None
        mul_tensor_1: "f32[256, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[512, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_default_5: "bf16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[256, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[512, 1]" = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_default_7: "bf16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[256, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f32[256, 512, 4, 4]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        gt_scalar: "b8[256, 512, 4, 4]" = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0)
        mul_tensor_3: "f32[256, 512, 4, 4]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.2)
        where_self: "f32[256, 512, 4, 4]" = torch.ops.aten.where.self(gt_scalar, convert_element_type_default_2, mul_tensor_3);  gt_scalar = convert_element_type_default_2 = mul_tensor_3 = None
        convert_element_type_default_3: "bf16[256, 512, 4, 4]" = torch.ops.prims.convert_element_type.default(where_self, torch.bfloat16);  where_self = None
        return convert_element_type_default_3


def _default_make_inputs():
    return [
    torch.randn([512], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512], dtype=torch.bfloat16, device='cuda'),
    torch.randn([256, 512, 4, 4], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512], dtype=torch.bfloat16, device='cuda'),
    torch.randn([512], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
