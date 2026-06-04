"""
Standalone repro captured via capture_hook.
Label: vllm_facebook_opt-125m_001
Pattern hash: a5d52cc24f84
Shape hash: f05d8499
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4, 512], i64), T([2050, 768], f16))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 512]", arg1_1: "f16[2050, 768]"):
        # No stacktrace found for following nodes
        add_tensor: "i64[4, 512]" = torch.ops.aten.add.Tensor(arg0_1, 2);  arg0_1 = None
        embedding_default: "f16[4, 512, 768]" = torch.ops.aten.embedding.default(arg1_1, add_tensor);  arg1_1 = add_tensor = None
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
