"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train_000
Pattern hash: 5e8fb91fe0df
Shape hash: 8fd18384
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([50265, 1024], f32), T([8, 1024], i64, gen=Index(50265)))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[50265, 1024]", arg1_1: "i64[8, 1024]"):
        # No stacktrace found for following nodes
        embedding_default: "f32[8, 1024, 1024]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 1);  arg0_1 = arg1_1 = None
        mul_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None
        return mul_tensor

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
