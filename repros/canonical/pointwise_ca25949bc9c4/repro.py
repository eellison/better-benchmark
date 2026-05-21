"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train
Pattern hash: ca25949bc9c4
Shape hash: d4e37485
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
_shapes_config = "(T([8, 4096, 512], f32))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[8, 4096, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:1780 in forward, code: hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        slice_tensor: "f32[8, 4096, 256]" = torch.ops.aten.slice.Tensor(tangents_1, 2, 0, 256)
        slice_tensor_1: "f32[8, 4096, 256]" = torch.ops.aten.slice.Tensor(tangents_1, 2, 256, 512);  tangents_1 = None
        add_tensor: "f32[8, 4096, 256]" = torch.ops.aten.add.Tensor(slice_tensor, slice_tensor_1);  slice_tensor = slice_tensor_1 = None
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
