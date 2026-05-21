"""
Standalone repro captured via capture_hook.
Label: hf_BartForCausalLM_train_003
Pattern hash: 1c5b4956323d
Shape hash: 1009e89b
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
_shapes_config = "(T([1024], i64), T([1026, 1024], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[1024]", arg1_1: "f32[1026, 1024]"):
        # No stacktrace found for following nodes
        unsqueeze_default: "i64[1, 1024]" = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        add_tensor: "i64[1, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_default, 2);  unsqueeze_default = None
        embedding_default: "f32[1, 1024, 1024]" = torch.ops.aten.embedding.default(arg1_1, add_tensor);  arg1_1 = add_tensor = None
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
