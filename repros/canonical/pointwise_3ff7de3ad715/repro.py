"""
Standalone repro captured via capture_hook.
Label: hf_M2M100ForConditionalGeneration_inference
Pattern hash: 3ff7de3ad715
Shape hash: d0a75f23
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_190: "f32[1024, 4096]", _shape_param_0, _shape_param_1, arg508_1: "f32[1024, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:471 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        reshape_default: "f32[8, 128, 4096]" = torch.ops.aten.reshape.default(addmm_190, _shape_param_0);  addmm_190 = _shape_param_0 = None
        relu_default: "f32[8, 128, 4096]" = torch.ops.aten.relu.default(reshape_default);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/m2m_100/modeling_m2m_100.py:473 in forward, code: hidden_states = self.fc2(hidden_states)
        reshape_default_1: "f32[1024, 4096]" = torch.ops.aten.reshape.default(relu_default, _shape_param_1);  relu_default = _shape_param_1 = None
        permute_default: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg508_1, [1, 0]);  arg508_1 = None
        return (reshape_default_1, permute_default)


def _default_make_inputs():
    return [
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    [8, 128, 4096],  # _shape_param_0
    [1024, 4096],  # _shape_param_1
    torch.randn([1024, 4096], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
