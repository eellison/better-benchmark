"""
Standalone repro captured via capture_hook.
Label: torchbench_dlrm_infer_000
Pattern hash: 3cfc4c59af74
Shape hash: 40d0a0c4
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
_shapes_config = "(T([36], i64, gen=Index(9)), T([36], i64, gen=Index(9)), T([2048, 9, 9], f32), T([2048, 64], f32))"

class Repro(torch.nn.Module):
    def forward(self, _tensor_constant0: "i64[36]", _tensor_constant1: "i64[36]", bmm: "f32[2048, 9, 9]", relu_1: "f32[2048, 64]"):
        # No stacktrace found for following nodes
        lift_fresh_copy_default: "i64[36]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        lift_fresh_copy_default_1: "i64[36]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        index_tensor: "f32[2048, 36]" = torch.ops.aten.index.Tensor(bmm, [None, lift_fresh_copy_default, lift_fresh_copy_default_1]);  bmm = lift_fresh_copy_default = lift_fresh_copy_default_1 = None
        cat_default: "f32[2048, 100]" = torch.ops.aten.cat.default([relu_1, index_tensor], 1);  relu_1 = index_tensor = None
        return cat_default



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
