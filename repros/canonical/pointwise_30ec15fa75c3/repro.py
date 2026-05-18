"""
Standalone repro captured via capture_hook.
Label: genai_gqa_decode
Pattern hash: 30ec15fa75c3
Shape hash: d2d094a8
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
    def forward(self, arg2_1: "bf16[32, 32, 1, 128]", _shape_param_0, _shape_param_1, arg0_1: "bf16[32, 8, 1, 128]", _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:147 in gqa_pattern, code: scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
        expand_default: "bf16[32, 32, 1, 128]" = torch.ops.aten.expand.default(arg2_1, _shape_param_0);  arg2_1 = _shape_param_0 = None
        reshape_default: "bf16[1024, 1, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_1);  expand_default = _shape_param_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:142 in gqa_pattern, code: K_expanded = K[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, hd)
        unsqueeze_default: "bf16[32, 8, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        expand_default_1: "bf16[32, 8, 4, 1, 128]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_2);  unsqueeze_default = _shape_param_2 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:143 in gqa_pattern, code: K_expanded = K_expanded.reshape(batch, num_kv_heads * n_rep, slen, hd)
        clone_default: "bf16[32, 8, 4, 1, 128]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_1: "bf16[32, 32, 1, 128]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_varied_shapes.py:147 in gqa_pattern, code: scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
        permute_default: "bf16[32, 32, 128, 1]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None
        expand_default_2: "bf16[32, 32, 128, 1]" = torch.ops.aten.expand.default(permute_default, _shape_param_4);  permute_default = _shape_param_4 = None
        reshape_default_2: "bf16[1024, 128, 1]" = torch.ops.aten.reshape.default(expand_default_2, _shape_param_5);  expand_default_2 = _shape_param_5 = None
        return (reshape_default, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([32, 32, 1, 128], dtype=torch.bfloat16, device='cuda'),
    [32, 32, 1, 128],  # _shape_param_0
    [1024, 1, 128],  # _shape_param_1
    torch.randn([32, 8, 1, 128], dtype=torch.bfloat16, device='cuda'),
    [32, 8, 4, 1, 128],  # _shape_param_2
    [32, 32, 1, 128],  # _shape_param_3
    [32, 32, 128, 1],  # _shape_param_4
    [1024, 128, 1],  # _shape_param_5
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
