"""
Standalone repro captured via capture_hook.
Label: genai_patterns
Pattern hash: 9d5fcd3a6ed4
Shape hash: e61d44f6
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4, 8, 2048, 128]", _shape_param_0, _shape_param_1, arg1_1: "bf16[4, 8, 2048, 128]", _shape_param_2, _shape_param_3):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:128 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(
        unsqueeze_default: "bf16[4, 8, 1, 2048, 128]" = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        expand_default: "bf16[4, 8, 4, 2048, 128]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:131 in repeat_kv, code: return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
        clone_default: "bf16[4, 8, 4, 2048, 128]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default: "bf16[4, 32, 2048, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:128 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(
        unsqueeze_default_1: "bf16[4, 8, 1, 2048, 128]" = torch.ops.aten.unsqueeze.default(arg1_1, 2);  arg1_1 = None
        expand_default_1: "bf16[4, 8, 4, 2048, 128]" = torch.ops.aten.expand.default(unsqueeze_default_1, _shape_param_2);  unsqueeze_default_1 = _shape_param_2 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:131 in repeat_kv, code: return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
        clone_default_1: "bf16[4, 8, 4, 2048, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_1: "bf16[4, 32, 2048, 128]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None
        return (reshape_default, reshape_default_1)


def _default_make_inputs():
    return [
    torch.randn([4, 8, 2048, 128], dtype=torch.bfloat16, device='cuda'),
    [4, 8, 4, 2048, 128],  # _shape_param_0
    [4, 32, 2048, 128],  # _shape_param_1
    torch.randn([4, 8, 2048, 128], dtype=torch.bfloat16, device='cuda'),
    [4, 8, 4, 2048, 128],  # _shape_param_2
    [4, 32, 2048, 128],  # _shape_param_3
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
