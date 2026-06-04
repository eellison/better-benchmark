"""
Standalone repro captured via capture_hook.
Label: hf_BlenderbotForCausalLM_train_003
Pattern hash: edf2e0dc44ea
Shape hash: d999c84a
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([128, 2560], f32), T([128], i64, gen=Index(128)))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f32[128, 2560]", arg0_1: "i64[128]"):
        # No stacktrace found for following nodes
        embedding_default: "f32[128, 2560]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
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
