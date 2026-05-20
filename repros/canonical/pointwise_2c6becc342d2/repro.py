"""
Standalone repro captured via capture_hook.
Label: timm_nfnet_l0_train
Pattern hash: 2c6becc342d2
Shape hash: 293d0a77
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 256, 56, 56], f32, stride=(802816, 1, 14336, 256)), T([128, 256, 1, 1], f32), T([128, 256, 1, 1], f32), S([128, 256, 56, 56]))"

class Repro(torch.nn.Module):
    def forward(self, mul_957: "f32[128, 256, 56, 56]", sigmoid: "f32[128, 256, 1, 1]", getitem_327: "f32[128, 256, 1, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[128, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_957, sigmoid);  mul_957 = sigmoid = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[128, 256, 56, 56]" = torch.ops.aten.expand.default(getitem_327, _shape_param_0);  getitem_327 = _shape_param_0 = None
        div_scalar: "f32[128, 256, 56, 56]" = torch.ops.aten.div.Scalar(expand_default, 3136);  expand_default = None
        add_tensor: "f32[128, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None
        return add_tensor


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
