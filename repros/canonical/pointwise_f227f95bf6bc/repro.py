"""
Standalone repro captured via capture_hook.
Label: genai_patterns
Pattern hash: f227f95bf6bc
Shape hash: 4f537d16
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[4, 32, 2048, 128]", arg1_1: "bf16[1, 1, 2048, 128]", arg2_1: "bf16[1, 1, 2048, 128]"):
        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:225 in rotary_embedding_pattern, code: return x * cos + rotated * sin
        mul_tensor: "bf16[4, 32, 2048, 128]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg1_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:222 in rotary_embedding_pattern, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_tensor: "bf16[4, 32, 2048, 64]" = torch.ops.aten.slice.Tensor(arg0_1, 3, 64, 9223372036854775807)

        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:224 in rotary_embedding_pattern, code: rotated = torch.cat((-x2, x1), dim=-1)
        neg_default: "bf16[4, 32, 2048, 64]" = torch.ops.aten.neg.default(slice_tensor);  slice_tensor = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:221 in rotary_embedding_pattern, code: x1 = x[..., : x.shape[-1] // 2]
        slice_tensor_1: "bf16[4, 32, 2048, 64]" = torch.ops.aten.slice.Tensor(arg0_1, 3, 0, 64);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:224 in rotary_embedding_pattern, code: rotated = torch.cat((-x2, x1), dim=-1)
        cat_default: "bf16[4, 32, 2048, 128]" = torch.ops.aten.cat.default([neg_default, slice_tensor_1], -1);  neg_default = slice_tensor_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_genai_patterns.py:225 in rotary_embedding_pattern, code: return x * cos + rotated * sin
        mul_tensor_1: "bf16[4, 32, 2048, 128]" = torch.ops.aten.mul.Tensor(cat_default, arg2_1);  cat_default = arg2_1 = None
        add_tensor: "bf16[4, 32, 2048, 128]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        return add_tensor


def _default_make_inputs():
    return [
    torch.randn([4, 32, 2048, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1, 2048, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1, 2048, 128], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
