"""
Standalone repro captured via capture_hook.
Label: timm_deit_base_distilled_patch16_224_training
Pattern hash: fb0528df1e3b
Shape hash: 27087f8d
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
    def forward(self, tangents_1: "f32[32, 1000]", primals_155: "f32[1000, 768]", primals_153: "f32[1000, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:124 in forward_head, code: return (x + x_dist) / 2
        div_tensor: "f32[32, 1000]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:118 in forward_head, code: x_dist = self.head_dist(x_dist)
        permute_default: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        permute_default_1: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/deit.py:117 in forward_head, code: x = self.head(x)
        permute_default_2: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        permute_default_3: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (div_tensor, permute_default_1, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([32, 1000], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
