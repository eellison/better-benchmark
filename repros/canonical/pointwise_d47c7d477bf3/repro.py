"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_infer
Pattern hash: d47c7d477bf3
Shape hash: 7665b346
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
_shapes_config = "(T([256, 1280], f16))"

class Repro(torch.nn.Module):
    def forward(self, addmm: "f16[256, 1280]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:216 in _forward_impl, code: x = self.classifier(x)
        convert_element_type_default: "f32[256, 1280]" = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        add_tensor: "f32[256, 1280]" = torch.ops.aten.add.Tensor(convert_element_type_default, 3)
        clamp_min_default: "f32[256, 1280]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[256, 1280]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor: "f32[256, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_default, clamp_max_default);  convert_element_type_default = clamp_max_default = None
        div_tensor: "f32[256, 1280]" = torch.ops.aten.div.Tensor(mul_tensor, 6);  mul_tensor = None
        convert_element_type_default_1: "f16[256, 1280]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None
        return convert_element_type_default_1



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
