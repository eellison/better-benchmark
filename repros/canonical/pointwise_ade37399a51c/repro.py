"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer_006
Pattern hash: ade37399a51c
Shape hash: b857f3e4
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
_shapes_config = "(T([32], i64, gen=Index(32)), T([32, 3, 224, 224], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[32]", arg1_1: "f32[32, 3, 224, 224]"):
        # No stacktrace found for following nodes
        view_default: "i64[1, 32]" = torch.ops.aten.view.default(arg0_1, [1, -1]);  arg0_1 = None
        select_int: "i64[32]" = torch.ops.aten.select.int(view_default, 0, 0);  view_default = None
        index_tensor: "f32[32, 3, 224, 224]" = torch.ops.aten.index.Tensor(arg1_1, [select_int]);  arg1_1 = select_int = None
        return index_tensor



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
