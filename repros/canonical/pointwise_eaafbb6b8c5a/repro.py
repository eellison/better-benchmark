"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer_006
Pattern hash: eaafbb6b8c5a
Shape hash: b80d2fb9
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
_shapes_config = "(T([32], i64))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[32]"):
        # No stacktrace found for following nodes
        sort_default = torch.ops.aten.sort.default(arg0_1);  arg0_1 = None
        getitem: "i64[32]" = sort_default[0]
        getitem_1: "i64[32]" = sort_default[1];  sort_default = None
        return (getitem, getitem_1)



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
