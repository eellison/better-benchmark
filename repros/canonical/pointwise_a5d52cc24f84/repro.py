"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: a5d52cc24f84
Shape hash: fc4d30c5
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
    def forward(self, primals_1: "i64[4, 2048]", primals_2: "f32[2050, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:70 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[4, 2048]" = torch.ops.aten.add.Tensor(primals_1, 2);  primals_1 = None
        embedding_default: "f32[4, 2048, 768]" = torch.ops.aten.embedding.default(primals_2, add_tensor);  primals_2 = add_tensor = None
        return embedding_default


def _default_make_inputs():
    return [
    torch.randint(0, 2048, [4, 2048], dtype=torch.int64, device='cuda'),
    torch.randn([2050, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
