"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/modeling_utils.py:1649 in get_extended_attention_mask, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        full_default: "f32[8, 1, 1, 1024]" = torch.ops.aten.full.default([8, 1, 1, 1024], -0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1591 in forward, code: extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[
        select_int: "f32[8, 1, 1024]" = torch.ops.aten.select.int(full_default, 1, 0);  full_default = None
        select_int_1: "f32[8, 1024]" = torch.ops.aten.select.int(select_int, 1, 0);  select_int = None
        return select_int_1


def _default_make_inputs():
    return [

    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
