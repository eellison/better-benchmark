"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_infer
Pattern hash: 26086f65d617
Shape hash: 98ec67b1
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
_shapes_config = "(T([128, 768, 12, 12], f32, stride=(110592, 1, 9216, 768)))"

class Repro(torch.nn.Module):
    def forward(self, convolution_62: "f32[128, 768, 12, 12]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:135 in gelu, code: return F.gelu(x)
        mul_tensor: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_62, 0.5)
        mul_tensor_1: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_62, 0.7071067811865476);  convolution_62 = None
        erf_default: "f32[128, 768, 12, 12]" = torch.ops.aten.erf.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor: "f32[128, 768, 12, 12]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_2: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:89 in forward, code: return self.act_fn(x, inplace=self.inplace).mul_(self.gamma)
        mul_tensor_3: "f32[128, 768, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.7015043497085571);  mul_tensor_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[128, 768, 13, 13]" = torch.ops.aten.constant_pad_nd.default(mul_tensor_3, [0, 1, 0, 1], 0.0);  mul_tensor_3 = None
        return constant_pad_nd_default



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
