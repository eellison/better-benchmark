"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-2-5-linux.aws.a100_graph4
Pattern hash: 57c8afbf3cee
Shape hash: 81f90883
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[512, 768]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None
        native_dropout_default = torch.ops.aten.native_dropout.default(view_default, 0.1, True);  view_default = None
        getitem: "f32[1, 512, 768]" = native_dropout_default[0]
        getitem_1: "b8[1, 512, 768]" = native_dropout_default[1];  native_dropout_default = None
        convert_element_type_default: "c64[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem, torch.complex64);  getitem = None
        return (convert_element_type_default, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    [1, 512, 768],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
