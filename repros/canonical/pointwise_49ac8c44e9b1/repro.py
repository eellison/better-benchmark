"""
Standalone repro captured via capture_hook.
Label: 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_train
Pattern hash: 49ac8c44e9b1
Shape hash: 12ea8e76
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 2560], f32))"

class Repro(torch.nn.Module):
    def forward(self, primals_1: "f32[128, 2560]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:79 in forward, code: position_ids = torch.arange(
        iota_default: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding_default: "f32[128, 2560]" = torch.ops.aten.embedding.default(primals_1, iota_default);  primals_1 = iota_default = None
        return embedding_default


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
