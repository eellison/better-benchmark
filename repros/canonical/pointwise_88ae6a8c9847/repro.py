"""
Standalone repro captured via capture_hook.
Label: timm_adv_inception_v3_train
Pattern hash: 88ae6a8c9847
Shape hash: 03818165
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
_shapes_config = "(T([1, 64, 1, 1], f32), T([64], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_5: "f32[1, 64, 1, 1]", primals_16: "f32[64]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        mul_tensor: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_tensor: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # No stacktrace found for following nodes
        copy__default: "f32[64]" = torch.ops.aten.copy_.default(primals_16, add_tensor);  primals_16 = add_tensor = None
        return copy__default



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
