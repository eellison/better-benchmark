"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_inference
Pattern hash: 263bc4201ad4
Shape hash: e866b3ad
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
    def forward(self, getitem_171: "f32[32, 32, 49, 32]", _shape_param_0, _shape_param_1, getitem_172: "f32[32, 32, 49, 32]", _shape_param_2, _shape_param_3):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_tensor: "f32[32, 32, 49, 32]" = torch.ops.aten.mul.Tensor(getitem_171, 0.1767766952966369);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        expand_default: "f32[32, 32, 49, 32]" = torch.ops.aten.expand.default(mul_tensor, _shape_param_0);  mul_tensor = _shape_param_0 = None
        clone_default: "f32[32, 32, 49, 32]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default: "f32[1024, 49, 32]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None
        permute_default: "f32[32, 32, 32, 49]" = torch.ops.aten.permute.default(getitem_172, [0, 1, 3, 2]);  getitem_172 = None
        expand_default_1: "f32[32, 32, 32, 49]" = torch.ops.aten.expand.default(permute_default, _shape_param_2);  permute_default = _shape_param_2 = None
        clone_default_1: "f32[32, 32, 32, 49]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_1: "f32[1024, 32, 49]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None
        return (reshape_default, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randn(4814848, dtype=torch.float32, device='cuda').as_strided([32, 32, 49, 32], [150528, 32, 3072, 1]),  # getitem_171
    [32, 32, 49, 32],  # _shape_param_0
    [1024, 49, 32],  # _shape_param_1
    torch.randn(4814848, dtype=torch.float32, device='cuda').as_strided([32, 32, 49, 32], [150528, 32, 3072, 1]),  # getitem_172
    [32, 32, 32, 49],  # _shape_param_2
    [1024, 32, 49],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
