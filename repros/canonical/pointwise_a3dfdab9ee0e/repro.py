"""
Standalone repro captured via capture_hook.
Label: torchbench_vgg16_train
Pattern hash: a3dfdab9ee0e
Shape hash: 297a9ccc
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 4096], b8), T([128, 4096], f32), T([128, 4096], b8))"

class Repro(torch.nn.Module):
    def forward(self, gt_1: "b8[128, 4096]", mm: "f32[128, 4096]", le: "b8[128, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        convert_element_type_default: "f32[128, 4096]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 2.0);  convert_element_type_default = None
        mul_tensor_1: "f32[128, 4096]" = torch.ops.aten.mul.Tensor(mm, mul_tensor);  mm = mul_tensor = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[128, 4096]" = torch.ops.aten.where.self(le, full_default, mul_tensor_1);  le = full_default = mul_tensor_1 = None
        return where_self



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
