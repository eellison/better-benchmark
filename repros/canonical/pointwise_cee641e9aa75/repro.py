"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_infer
Pattern hash: cee641e9aa75
Shape hash: ee32fdb4
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([], i64))"

class Repro(torch.nn.Module):
    def forward(self, gcd: "i64[]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:167 in _get_least_common_mult_chunk_len, code: return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
        eq_scalar: "b8[]" = torch.ops.aten.eq.Scalar(gcd, 0);  gcd = None
        return eq_scalar



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
