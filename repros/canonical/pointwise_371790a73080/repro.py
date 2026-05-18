"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 512]", arg1_1: "f16[2050, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:76 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[4, 512]" = torch.ops.aten.add.Tensor(arg0_1, 2);  arg0_1 = None
        embedding_default: "f16[4, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, add_tensor);  arg1_1 = add_tensor = None
        return embedding_default


def _default_make_inputs():
    return [
    torch.randint(0, 2, [4, 512], dtype=torch.int64, device='cuda'),
    torch.randn([2050, 768], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
