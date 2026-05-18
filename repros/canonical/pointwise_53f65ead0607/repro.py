"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train
Pattern hash: 53f65ead0607
Shape hash: 5c4001a3
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, primals_2: "f32[50265, 1024]", primals_1: "f32[8, 1024, 1024]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:1294 in torch_dynamo_resume_in_forward_at_1280, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_default: "f32[1024, 50265]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        reshape_default: "f32[8192, 1024]" = torch.ops.aten.reshape.default(primals_1, _shape_param_0);  primals_1 = _shape_param_0 = None
        constant_pad_nd_default: "f32[1024, 50268]" = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 3, 0, 0]);  permute_default = None
        return (reshape_default, constant_pad_nd_default)


def _default_make_inputs():
    return [
    torch.randn([50265, 1024], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1024], dtype=torch.float32, device='cuda'),
    [8192, 1024],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
