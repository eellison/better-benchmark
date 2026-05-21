"""
Standalone repro captured via capture_hook.
Label: torchbench_vgg16_train
Pattern hash: 103e6e2b1157
Shape hash: 5c93fbcd
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
    def forward(self, gt_1: "b8[128, 4096]", mm: "f32[128, 4096]", le: "b8[128, 4096]", primals_30: "f32[4096, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        convert_element_type_default: "f32[128, 4096]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 2.0);  convert_element_type_default = None
        mul_tensor_1: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(mm, mul_tensor);  mm = mul_tensor = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 4096]" = torch.ops.aten.where.self(le, full_default, mul_tensor_1);  le = full_default = mul_tensor_1 = None
        permute_default: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_default_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (where_self, permute_default_1)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [128, 4096], dtype=torch.bool, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [128, 4096], dtype=torch.bool, device='cuda'),
    torch.randn([4096, 4096], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
