"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_inference
Pattern hash: fa3f1bab3fdf
Shape hash: 9b7fe6ae
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
    def forward(self, arg0_1: "f32[8, 1024, 768]", _shape_param_0, arg1_1: "f32[768, 768]", _shape_param_1, arg3_1: "f32[768, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_default: "f32[1024, 8, 768]" = torch.ops.aten.permute.default(arg0_1, [1, 0, 2]);  arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        clone_default: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format)
        reshape_default: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_0);  clone_default = _shape_param_0 = None
        permute_default_1: "f32[768, 768]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        clone_default_1: "f32[1024, 8, 768]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_1);  clone_default_1 = _shape_param_1 = None
        permute_default_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        return (reshape_default, permute_default_1, reshape_default_1, permute_default_2)


def _default_make_inputs():
    return [
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    [8192, 768],  # _shape_param_0
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    [8192, 768],  # _shape_param_1
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
