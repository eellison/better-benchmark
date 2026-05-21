"""
Standalone repro captured via capture_hook.
Label: timm_convnextv2_nano.fcmae_ft_in22k_in1k_train
Pattern hash: 7e39816638f7
Shape hash: 04f73e4d
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
_shapes_config = "(T([128, 1, 1, 640], f32), T([128, 1, 1, 640], f32))"

class Repro(torch.nn.Module):
    def forward(self, permute_43: "f32[128, 1, 1, 640]", mul_92: "f32[128, 1, 1, 640]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_tensor: "f32[128, 1, 1, 640]" = torch.ops.aten.mul.Tensor(permute_43, mul_92);  permute_43 = mul_92 = None
        sum_dim_int_list: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1, 2]);  mul_tensor = None
        return sum_dim_int_list



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
