"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_infer_001
Pattern hash: 36d7a6209c54
Shape hash: 8f4d0096
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([], i32, gen=Index(6144)), T([], i32, gen=Index(6144)), T([6144], i64, gen=Index(6144)))"

class Repro(torch.nn.Module):
    def forward(self, arg2_1: "i32[]", arg3_1: "i32[]", arg4_1: "i64[6144]"):
        # No stacktrace found for following nodes
        ge_tensor: "b8[]" = torch.ops.aten.ge.Tensor(arg2_1, arg3_1)
        index_tensor: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg2_1]);  arg2_1 = None
        index_tensor_1: "i64[]" = torch.ops.aten.index.Tensor(arg4_1, [arg3_1]);  arg4_1 = arg3_1 = None
        eq_tensor: "b8[]" = torch.ops.aten.eq.Tensor(index_tensor, index_tensor_1);  index_tensor = index_tensor_1 = None
        bitwise_and_tensor: "b8[]" = torch.ops.aten.bitwise_and.Tensor(ge_tensor, eq_tensor);  ge_tensor = eq_tensor = None
        return bitwise_and_tensor

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
