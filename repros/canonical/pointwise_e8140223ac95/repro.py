"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: e8140223ac95
Shape hash: 00f9d69c
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
    def forward(self, gt_1: "b8[1024, 4096]", mm_2: "f32[1024, 4096]", le_1: "b8[1024, 4096]", full_default: "f32[]", primals_12: "f32[4096, 9216]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        convert_element_type_default: "f32[1024, 4096]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor: "f32[1024, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 2.0);  convert_element_type_default = None
        mul_tensor_1: "f32[1024, 4096]" = torch.ops.aten.mul.Tensor(mm_2, mul_tensor);  mm_2 = mul_tensor = None
        where_self: "f32[1024, 4096]" = torch.ops.aten.where.self(le_1, full_default, mul_tensor_1);  le_1 = full_default = mul_tensor_1 = None
        permute_default: "f32[9216, 4096]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_default_1: "f32[4096, 9216]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (where_self, permute_default_1)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [1024, 4096], dtype=torch.bool, device='cuda'),
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [1024, 4096], dtype=torch.bool, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([4096, 9216], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
