"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForQuestionAnswering_inference
Pattern hash: 0b16190d48f9
Shape hash: c2d5e989
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_56: "f32[128, 2]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:815 in forward, code: logits = self.qa_outputs(sequence_output)
        reshape_default: "f32[1, 128, 2]" = torch.ops.aten.reshape.default(addmm_56, _shape_param_0);  addmm_56 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:816 in forward, code: start_logits, end_logits = logits.split(1, dim=-1)
        split_tensor = torch.ops.aten.split.Tensor(reshape_default, 1, -1);  reshape_default = None
        getitem: "f32[1, 128, 1]" = split_tensor[0]
        getitem_1: "f32[1, 128, 1]" = split_tensor[1];  split_tensor = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([128, 2], dtype=torch.float32, device='cuda'),
    [1, 128, 2],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
