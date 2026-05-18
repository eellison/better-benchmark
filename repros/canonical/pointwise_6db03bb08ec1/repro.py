"""
Standalone repro captured via capture_hook.
Label: hf_TrOCRForCausalLM_training
Pattern hash: 6db03bb08ec1
Shape hash: 02575374
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
    def forward(self, _shape_param_0, primals_1: "f32[514, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:55 in forward, code: position_ids = torch.arange(
        iota_default: "i64[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:57 in forward, code: ).expand(bsz, -1)
        expand_default: "i64[8, 256]" = torch.ops.aten.expand.default(iota_default, _shape_param_0);  iota_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/trocr/modeling_trocr.py:61 in forward, code: return super().forward(position_ids + self.offset)
        add_tensor: "i64[8, 256]" = torch.ops.aten.add.Tensor(expand_default, 2);  expand_default = None
        embedding_default: "f32[8, 256, 1024]" = torch.ops.aten.embedding.default(primals_1, add_tensor);  primals_1 = add_tensor = None
        return embedding_default


def _default_make_inputs():
    return [
    [8, -1],  # _shape_param_0
    torch.randn([514, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
