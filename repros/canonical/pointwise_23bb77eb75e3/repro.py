"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_training
Pattern hash: 23bb77eb75e3
Shape hash: 71666bed
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
    def forward(self, gt: "b8[8, 1280]", mm: "f32[8, 1280]", _shape_param_0, le: "b8[8, 1280, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:836 in forward_head, code: x = F.dropout(x, p=self.drop_rate, training=self.training)
        convert_element_type_default: "f32[8, 1280]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[8, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.25);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1280]" = torch.ops.aten.mul.Tensor(mm, mul_tensor);  mm = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:834 in forward_head, code: x = self.flatten(x)
        reshape_default: "f32[8, 1280, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  mul_tensor_1 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:833 in forward_head, code: x = self.act2(x)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1280, 1, 1]" = torch.ops.aten.where.self(le, full_default, reshape_default);  le = full_default = reshape_default = None
        return where_self


def _default_make_inputs():
    return [
    torch.randint(0, 2, [8, 1280], dtype=torch.bool, device='cuda'),
    torch.randn([8, 1280], dtype=torch.float32, device='cuda'),
    [8, 1280, 1, 1],  # _shape_param_0
    torch.randint(0, 2, [8, 1280, 1, 1], dtype=torch.bool, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
