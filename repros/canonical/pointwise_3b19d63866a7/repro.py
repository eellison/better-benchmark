"""
Standalone repro captured via capture_hook.
Label: torchbench_moco_infer_007
Pattern hash: 3b19d63866a7
Shape hash: 5d885fec
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
_shapes_config = "(T([32], i64, gen=Index(32)), T([32, 128], f32), S([32, 128, 1]))"

class Repro(torch.nn.Module):
    def forward(self, arg268_1: "i64[32]", wait_tensor: "f32[32, 128]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "i64[1, 32]" = torch.ops.aten.view.default(arg268_1, [1, -1]);  arg268_1 = None
        select_int: "i64[32]" = torch.ops.aten.select.int(view_default, 0, 0);  view_default = None
        index_tensor: "f32[32, 128]" = torch.ops.aten.index.Tensor(wait_tensor, [select_int]);  wait_tensor = select_int = None
        view_default_1: "f32[32, 128, 1]" = torch.ops.aten.view.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None
        return view_default_1



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
