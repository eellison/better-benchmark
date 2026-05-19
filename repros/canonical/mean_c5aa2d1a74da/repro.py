"""
Standalone repro captured via capture_hook.
Label: inductor_timm_perf_cuda_h100-6-7-linux.aws.h100_graph58
Pattern hash: c5aa2d1a74da
Shape hash: 23cdf2ea
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
    def forward(self, arg274_1: "bf16[640]", arg275_1: "bf16[640]", convolution_34: "bf16[8, 640, 8, 8]", arg276_1: "bf16[640]", arg277_1: "bf16[640]", arg278_1: "bf16[1000, 640]", _shape_param_0):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[640]" = torch.ops.prims.convert_element_type.default(arg274_1, torch.float32);  arg274_1 = None
        convert_element_type_default_1: "f32[640]" = torch.ops.prims.convert_element_type.default(arg275_1, torch.float32);  arg275_1 = None
        add_tensor: "f32[640]" = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-05);  convert_element_type_default_1 = None
        sqrt_default: "f32[640]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[640]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[640]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_default, -1);  convert_element_type_default = None
        unsqueeze_default_1: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        unsqueeze_default_2: "f32[640, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        sub_tensor: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_default_1);  convolution_34 = unsqueeze_default_1 = None
        mul_tensor_1: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "bf16[640, 1]" = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_default_5: "bf16[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "bf16[640, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_default_7: "bf16[640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None
        convert_element_type_default_2: "f32[8, 640, 8, 8]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        neg_default: "f32[8, 640, 8, 8]" = torch.ops.aten.neg.default(convert_element_type_default_2)
        exp_default: "f32[8, 640, 8, 8]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_2: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[8, 640, 8, 8]" = torch.ops.aten.div.Tensor(convert_element_type_default_2, add_tensor_2);  convert_element_type_default_2 = add_tensor_2 = None
        convert_element_type_default_3: "bf16[8, 640, 8, 8]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        mean_dim: "bf16[8, 640, 1, 1]" = torch.ops.aten.mean.dim(convert_element_type_default_3, [-1, -2], True);  convert_element_type_default_3 = None
        view_default: "bf16[8, 640]" = torch.ops.aten.view.default(mean_dim, _shape_param_0);  mean_dim = _shape_param_0 = None
        permute_default: "bf16[640, 1000]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        return (view_default, permute_default)


def _default_make_inputs():
    return [
    torch.randn([640], dtype=torch.bfloat16, device='cuda'),
    torch.randn([640], dtype=torch.bfloat16, device='cuda'),
    torch.randn([8, 640, 8, 8], dtype=torch.bfloat16, device='cuda'),
    torch.randn([640], dtype=torch.bfloat16, device='cuda'),
    torch.randn([640], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1000, 640], dtype=torch.bfloat16, device='cuda'),
    [8, 640],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
