"""
Standalone repro captured via capture_hook.
Label: torchbench_dcgan_train
Pattern hash: 64ab03ebcf82
Shape hash: 1e61b242
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
    def forward(self, convolution_3: "f32[1024, 512, 4, 4]", primals_19: "f32[512]", primals_20: "f32[512]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/dcgan/__init__.py:128 in forward, code: return self.main(input)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 512, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 512, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 512, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 512, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1024, 512, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_3, getitem_1);  convolution_3 = getitem_1 = None
        mul_tensor: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_default_1: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_20, -1);  primals_20 = None
        unsqueeze_default_3: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[1024, 512, 4, 4]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        gt_scalar: "b8[1024, 512, 4, 4]" = torch.ops.aten.gt.Scalar(add_tensor_1, 0)
        mul_tensor_2: "f32[1024, 512, 4, 4]" = torch.ops.aten.mul.Tensor(add_tensor_1, 0.2)
        where_self: "f32[1024, 512, 4, 4]" = torch.ops.aten.where.self(gt_scalar, add_tensor_1, mul_tensor_2);  gt_scalar = add_tensor_1 = mul_tensor_2 = None
        return where_self


def _default_make_inputs():
    return [
    torch.randn([1024, 512, 4, 4], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
