"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train
Pattern hash: 083b9a31540a
Shape hash: 7df6f7e9
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([1, 512], i64), T([1, 512], i64), T([30000, 128], f32), T([8, 512], i64), T([2, 128], f32), T([512, 128], f32), T([128], f32), T([128], f32), T([4096, 128], f32), S([8, 512]), S([4096, 128]))"

class Repro(torch.nn.Module):
    def forward(self, primals_3: "i64[1, 512]", primals_2: "i64[1, 512]", primals_4: "f32[30000, 128]", primals_1: "i64[8, 512]", primals_5: "f32[2, 128]", primals_6: "f32[512, 128]", primals_7: "f32[128]", primals_8: "f32[128]", primals_9: "f32[4096, 128]", _shape_param_0, _shape_param_1):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py:94 in forward, code: buffered_token_type_ids = self.token_type_ids.expand(position_ids.shape[0], -1)
        expand_default: "i64[1, 512]" = torch.ops.aten.expand.default(primals_3, [1, -1]);  primals_3 = None


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
