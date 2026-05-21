"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_infer
Pattern hash: a72ccae4cfa7
Shape hash: 243e759c
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
_shapes_config = "(T([12000, 256], f16), S([8, 1500, 256]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_20: "f16[12000, 256]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1333 in forward, code: hidden_states = self.projector(hidden_states)
        reshape_default: "f16[8, 1500, 256]" = torch.ops.aten.reshape.default(addmm_20, _shape_param_0);  addmm_20 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1334 in forward, code: pooled_output = hidden_states.mean(dim=1)
        mean_dim: "f16[8, 256]" = torch.ops.aten.mean.dim(reshape_default, [1]);  reshape_default = None
        return mean_dim



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
