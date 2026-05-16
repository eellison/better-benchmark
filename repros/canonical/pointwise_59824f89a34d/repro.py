"""
Standalone repro captured via capture_hook.
Label: tritonbench_rope_B1_S2048_H32_D128
Pattern hash: 59824f89a34d
Shape hash: 44402fcc
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 32, 2048, 128]", arg1_1: "bf16[1, 1, 2048, 64]", arg2_1: "bf16[1, 1, 2048, 64]"):
        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:157 in rope_fn, code: q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
        slice_tensor: "bf16[1, 32, 2048, 64]" = torch.ops.aten.slice.Tensor(arg0_1, 3, 0, 64)

        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:158 in rope_fn, code: return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        mul_tensor: "bf16[1, 32, 2048, 64]" = torch.ops.aten.mul.Tensor(slice_tensor, arg1_1)

        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:157 in rope_fn, code: q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
        slice_tensor_1: "bf16[1, 32, 2048, 64]" = torch.ops.aten.slice.Tensor(arg0_1, 3, 64, 9223372036854775807);  arg0_1 = None

        # File: /tmp/scratch_space/better_benchmark/capture_tritonbench_ops.py:158 in rope_fn, code: return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        mul_tensor_1: "bf16[1, 32, 2048, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_1, arg2_1)
        sub_tensor: "bf16[1, 32, 2048, 64]" = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        mul_tensor_2: "bf16[1, 32, 2048, 64]" = torch.ops.aten.mul.Tensor(slice_tensor_1, arg1_1);  slice_tensor_1 = arg1_1 = None
        mul_tensor_3: "bf16[1, 32, 2048, 64]" = torch.ops.aten.mul.Tensor(slice_tensor, arg2_1);  slice_tensor = arg2_1 = None
        add_tensor: "bf16[1, 32, 2048, 64]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        cat_default: "bf16[1, 32, 2048, 128]" = torch.ops.aten.cat.default([sub_tensor, add_tensor], -1);  sub_tensor = add_tensor = None
        return cat_default


def _default_make_inputs():
    return [
    torch.randn([1, 32, 2048, 128], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1, 2048, 64], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1, 1, 2048, 64], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
