"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_training
Pattern hash: 663c8643c9af
Shape hash: 10a6b9c9
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
    def forward(self, mul_1381: "f32[32, 256, 48, 48]", sigmoid: "f32[32, 256, 1, 1]", getitem_327: "f32[32, 256, 1, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[32, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1381, sigmoid);  mul_1381 = sigmoid = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[32, 256, 48, 48]" = torch.ops.aten.expand.default(getitem_327, _shape_param_0);  getitem_327 = _shape_param_0 = None
        div_scalar: "f32[32, 256, 48, 48]" = torch.ops.aten.div.Scalar(expand_default, 2304);  expand_default = None
        add_tensor: "f32[32, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None
        return add_tensor


def _default_make_inputs():
    return [
    torch.randn([32, 256, 48, 48], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 256, 1, 1], dtype=torch.float32, device='cuda'),
    [32, 256, 48, 48],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
