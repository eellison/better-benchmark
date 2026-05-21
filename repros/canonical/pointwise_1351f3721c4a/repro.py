"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_train
Pattern hash: 1351f3721c4a
Shape hash: beb38d20
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
_shapes_config = "(T([256, 1280], b8), T([256, 1280], f32), T([256, 1280], f32))"

class Repro(torch.nn.Module):
    def forward(self, lt: "b8[256, 1280]", mm: "f32[256, 1280]", addmm: "f32[256, 1280]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:216 in _forward_impl, code: x = self.classifier(x)
        convert_element_type_default: "f32[256, 1280]" = torch.ops.prims.convert_element_type.default(lt, torch.float32);  lt = None
        div_scalar: "f32[256, 1280]" = torch.ops.aten.div.Scalar(convert_element_type_default, 0.8);  convert_element_type_default = None
        mul_tensor: "f32[256, 1280]" = torch.ops.aten.mul.Tensor(mm, div_scalar);  mm = div_scalar = None
        le_scalar: "b8[256, 1280]" = torch.ops.aten.le.Scalar(addmm, -3)
        lt_scalar: "b8[256, 1280]" = torch.ops.aten.lt.Scalar(addmm, 3)
        div_tensor: "f32[256, 1280]" = torch.ops.aten.div.Tensor(addmm, 3);  addmm = None
        add_tensor: "f32[256, 1280]" = torch.ops.aten.add.Tensor(div_tensor, 0.5);  div_tensor = None
        mul_tensor_1: "f32[256, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  add_tensor = None
        where_self: "f32[256, 1280]" = torch.ops.aten.where.self(lt_scalar, mul_tensor_1, mul_tensor);  lt_scalar = mul_tensor_1 = mul_tensor = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1: "f32[256, 1280]" = torch.ops.aten.where.self(le_scalar, full_default, where_self);  le_scalar = full_default = where_self = None
        return where_self_1



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
