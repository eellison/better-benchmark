"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g21
Pattern hash: 2c048e3bbc7a
Shape hash: 07b94259
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm_71: "f16[512, 768]"):
        # No stacktrace found for following nodes
        reshape_default: "f16[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_71, [1, 512, 768]);  addmm_71 = None
        native_dropout_default = torch.ops.aten.native_dropout.default(reshape_default, 0.1, True);  reshape_default = None
        getitem: "f16[1, 512, 768]" = native_dropout_default[0]
        getitem_1: "b8[1, 512, 768]" = native_dropout_default[1];  native_dropout_default = None
        return (getitem, getitem_1)


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
