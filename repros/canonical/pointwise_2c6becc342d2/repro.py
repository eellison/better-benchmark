"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train
Pattern hash: 2c6becc342d2
Shape hash: 0f41d4d2
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
_shapes_config = "(T([128, 256, 48, 48], f32, stride=(589824, 1, 12288, 256)), T([128, 256, 1, 1], f32), T([128, 256, 1, 1], f32), S([128, 256, 48, 48]))"

class Repro(torch.nn.Module):
    def forward(self, mul_1381: "f32[128, 256, 48, 48]", sigmoid: "f32[128, 256, 1, 1]", getitem_327: "f32[128, 256, 1, 1]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor: "f32[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1381, sigmoid);  mul_1381 = sigmoid = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_default: "f32[128, 256, 48, 48]" = torch.ops.aten.expand.default(getitem_327, _shape_param_0);  getitem_327 = _shape_param_0 = None
        div_scalar: "f32[128, 256, 48, 48]" = torch.ops.aten.div.Scalar(expand_default, 2304);  expand_default = None
        add_tensor: "f32[128, 256, 48, 48]" = torch.ops.aten.add.Tensor(mul_tensor, div_scalar);  mul_tensor = div_scalar = None
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
