"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: 9b83d9cb3773
Shape hash: 1c5202a8
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
_shapes_config = "(T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32), T([4096, 16384], f32))"

class Repro(torch.nn.Module):
    def forward(self, mm_5: "f32[4096, 16384]", mm_17: "f32[4096, 16384]", mm_29: "f32[4096, 16384]", mm_41: "f32[4096, 16384]", mm_53: "f32[4096, 16384]", mm_65: "f32[4096, 16384]", mm_77: "f32[4096, 16384]", mm_89: "f32[4096, 16384]", mm_101: "f32[4096, 16384]", mm_113: "f32[4096, 16384]", mm_125: "f32[4096, 16384]", mm_137: "f32[4096, 16384]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(mm_5, mm_17);  mm_5 = mm_17 = None
        add_tensor_1: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor, mm_29);  add_tensor = mm_29 = None
        add_tensor_2: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_1, mm_41);  add_tensor_1 = mm_41 = None
        add_tensor_3: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_2, mm_53);  add_tensor_2 = mm_53 = None
        add_tensor_4: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_3, mm_65);  add_tensor_3 = mm_65 = None
        add_tensor_5: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_4, mm_77);  add_tensor_4 = mm_77 = None
        add_tensor_6: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_5, mm_89);  add_tensor_5 = mm_89 = None
        add_tensor_7: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_6, mm_101);  add_tensor_6 = mm_101 = None
        add_tensor_8: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_7, mm_113);  add_tensor_7 = mm_113 = None
        add_tensor_9: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_8, mm_125);  add_tensor_8 = mm_125 = None
        add_tensor_10: "f32[4096, 16384]" = torch.ops.aten.add.Tensor(add_tensor_9, mm_137);  add_tensor_9 = mm_137 = None
        return add_tensor_10



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
