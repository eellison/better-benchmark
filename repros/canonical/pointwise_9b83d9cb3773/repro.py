"""
Standalone repro captured via capture_hook.
Label: hf_AlbertForMaskedLM_train_001
Pattern hash: 9b83d9cb3773
Shape hash: 35949c2c
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
_shapes_config = "(T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32))"

class Repro(torch.nn.Module):
    def forward(self, mm_15: "f32[4096, 4096]", mm_27: "f32[4096, 4096]", mm_39: "f32[4096, 4096]", mm_51: "f32[4096, 4096]", mm_63: "f32[4096, 4096]", mm_75: "f32[4096, 4096]", mm_87: "f32[4096, 4096]", mm_99: "f32[4096, 4096]", mm_111: "f32[4096, 4096]", mm_123: "f32[4096, 4096]", mm_135: "f32[4096, 4096]", mm_147: "f32[4096, 4096]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(mm_15, mm_27);  mm_15 = mm_27 = None
        add_tensor_1: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor, mm_39);  add_tensor = mm_39 = None
        add_tensor_2: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_1, mm_51);  add_tensor_1 = mm_51 = None
        add_tensor_3: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_2, mm_63);  add_tensor_2 = mm_63 = None
        add_tensor_4: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_3, mm_75);  add_tensor_3 = mm_75 = None
        add_tensor_5: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_4, mm_87);  add_tensor_4 = mm_87 = None
        add_tensor_6: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_5, mm_99);  add_tensor_5 = mm_99 = None
        add_tensor_7: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_6, mm_111);  add_tensor_6 = mm_111 = None
        add_tensor_8: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_7, mm_123);  add_tensor_7 = mm_123 = None
        add_tensor_9: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_8, mm_135);  add_tensor_8 = mm_135 = None
        add_tensor_10: "f32[4096, 4096]" = torch.ops.aten.add.Tensor(add_tensor_9, mm_147);  add_tensor_9 = mm_147 = None
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
