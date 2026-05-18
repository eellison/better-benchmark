"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForQuestionAnswering_inference
Pattern hash: 3568a3388902
Shape hash: 7b6b64f7
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
    def forward(self, getitem_114: "f32[1, 128, 1]", getitem_115: "f32[1, 128, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:817 in forward, code: start_logits = start_logits.squeeze(-1).contiguous()
        squeeze_dim: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_114, -1);  getitem_114 = None
        clone_default: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:818 in forward, code: end_logits = end_logits.squeeze(-1).contiguous()
        squeeze_dim_1: "f32[1, 128]" = torch.ops.aten.squeeze.dim(getitem_115, -1);  getitem_115 = None
        clone_default_1: "f32[1, 128]" = torch.ops.aten.clone.default(squeeze_dim_1, memory_format = torch.contiguous_format);  squeeze_dim_1 = None
        return (clone_default, clone_default_1)


def _default_make_inputs():
    return [
    torch.randn(255, dtype=torch.float32, device='cuda').as_strided([1, 128, 1], [256, 2, 1]),  # getitem_114
    torch.randn(255, dtype=torch.float32, device='cuda').as_strided([1, 128, 1], [256, 2, 1]),  # getitem_115
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
