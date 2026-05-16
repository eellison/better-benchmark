"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g21
Pattern hash: 3e6393c7164c
Shape hash: 21b18adb
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[1, 512, 1]", add_7: "f32[1, 512, 768]", getitem_1: "f32[1, 512, 1]", arg9_1: "f32[768]", arg10_1: "f32[768]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt_default: "f32[1, 512, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_1);  add_7 = getitem_1 = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg9_1);  mul_tensor = arg9_1 = None
        add_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_tensor_1, arg10_1);  mul_tensor_1 = arg10_1 = None
        native_dropout_default = torch.ops.aten.native_dropout.default(add_tensor_1, 0.1, True);  add_tensor_1 = None
        getitem_2: "f32[1, 512, 768]" = native_dropout_default[0]
        getitem_3: "b8[1, 512, 768]" = native_dropout_default[1];  native_dropout_default = None
        return (getitem_2, getitem_3)


def _default_make_inputs():
    return [
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
