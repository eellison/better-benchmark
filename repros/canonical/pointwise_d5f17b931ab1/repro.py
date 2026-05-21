"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_infer_000
Pattern hash: d5f17b931ab1
Shape hash: 0abf12fe
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
_shapes_config = "(T([16384, 768], f32), T([128, 128, 768], f32), S([128, 128, 768]), S([16384, 768]))"

class Repro(torch.nn.Module):
    def forward(self, addmm_71: "f32[16384, 768]", add_81: "f32[128, 128, 768]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[128, 128, 768]" = torch.ops.aten.view.default(addmm_71, _shape_param_0);  addmm_71 = _shape_param_0 = None
        add_tensor: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_81, view_default);  add_81 = view_default = None
        select_int: "f32[128, 768]" = torch.ops.aten.select.int(add_tensor, 1, 0)
        view_default_1: "f32[16384, 768]" = torch.ops.aten.view.default(add_tensor, _shape_param_1);  add_tensor = _shape_param_1 = None
        return (select_int, view_default_1)



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
