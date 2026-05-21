"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train
Pattern hash: 7a6fa515d427
Shape hash: 5af27dfa
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
_shapes_config = "(T([8, 1500, 384], f32, stride=(576000, 1, 1500)), S([12000, 384]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[8, 1500, 384]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:405 in forward, code: hidden_states = self.fc2(hidden_states)
        clone_default: "f32[8, 1500, 384]" = torch.ops.aten.clone.default(tangents_1, memory_format = torch.contiguous_format);  tangents_1 = None
        reshape_default: "f32[12000, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        return reshape_default



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
