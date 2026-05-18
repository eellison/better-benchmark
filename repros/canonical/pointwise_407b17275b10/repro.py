"""
Standalone repro captured via capture_hook.
Label: hf_Reformer_inference
Pattern hash: 407b17275b10
Shape hash: d69025d3
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
    def forward(self, _shape_param_0, arg1_1: "f32[320, 256]", arg0_1: "i64[8, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:324 in forward, code: position_ids = torch.arange(
        iota_default: "i64[4096]" = torch.ops.prims.iota.default(4096, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:327 in forward, code: position_ids = position_ids.unsqueeze(0).expand(input_shape)
        unsqueeze_default: "i64[1, 4096]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None
        expand_default: "i64[8, 4096]" = torch.ops.aten.expand.default(unsqueeze_default, _shape_param_0);  unsqueeze_default = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:330 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding_default: "f32[8, 4096, 256]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        return (expand_default, embedding_default)


def _default_make_inputs():
    return [
    [8, 4096],  # _shape_param_0
    torch.randn([320, 256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 4096], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
