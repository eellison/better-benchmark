"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_inference
Pattern hash: 1a8cf288331e
Shape hash: ccfec8df
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_48: "f32[32, 1000]", addmm_49: "f32[32, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:124 in forward_head, code: return (x + x_dist) / 2
        add_tensor: "f32[32, 1000]" = torch.ops.aten.add.Tensor(addmm_48, addmm_49);  addmm_48 = addmm_49 = None
        div_tensor: "f32[32, 1000]" = torch.ops.aten.div.Tensor(add_tensor, 2);  add_tensor = None
        return div_tensor


def _default_make_inputs():
    return [
    torch.randn([32, 1000], dtype=torch.float32, device='cuda'),
    torch.randn([32, 1000], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
