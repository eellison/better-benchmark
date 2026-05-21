"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Whisper_train
Pattern hash: 1ff82fee95ff
Shape hash: 30e28b9a
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
_shapes_config = "(T([8, 256], f32), S([8, 1500, 256]), S([12000, 256]))"

class Repro(torch.nn.Module):
    def forward(self, mm: "f32[8, 256]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1334 in torch_dynamo_resume_in_forward_at_1314, code: pooled_output = hidden_states.mean(dim=1)
        unsqueeze_default: "f32[8, 1, 256]" = torch.ops.aten.unsqueeze.default(mm, 1);  mm = None
        expand_default: "f32[8, 1500, 256]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None
        div_scalar: "f32[8, 1500, 256]" = torch.ops.aten.div.Scalar(expand_default, 1500);  expand_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py:1333 in torch_dynamo_resume_in_forward_at_1314, code: hidden_states = self.projector(hidden_states)
        reshape_default: "f32[12000, 256]" = torch.ops.aten.reshape.default(div_scalar, _shape_param_1);  div_scalar = _shape_param_1 = None
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
