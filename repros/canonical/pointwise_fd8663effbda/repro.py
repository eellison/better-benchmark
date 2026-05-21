"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_infer
Pattern hash: fd8663effbda
Shape hash: 407e8d49
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
    def forward(self, convolution_59: "f16[256, 960, 1, 1]", convert_element_type_183: "f16[256, 960, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:257 in _scale, code: return self.scale_activation(scale)
        convert_element_type_default: "f32[256, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32);  convolution_59 = None
        add_tensor: "f32[256, 960, 1, 1]" = torch.ops.aten.add.Tensor(convert_element_type_default, 3);  convert_element_type_default = None
        clamp_min_default: "f32[256, 960, 1, 1]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[256, 960, 1, 1]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        div_tensor: "f32[256, 960, 1, 1]" = torch.ops.aten.div.Tensor(clamp_max_default, 6);  clamp_max_default = None
        convert_element_type_default_1: "f16[256, 960, 1, 1]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:261 in forward, code: return scale * input
        mul_tensor: "f16[256, 960, 7, 7]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_183);  convert_element_type_default_1 = convert_element_type_183 = None
        return mul_tensor


def _default_make_inputs():
    return [
    torch.randn([256, 960, 1, 1], dtype=torch.float16, device='cuda'),
    torch.randn([256, 960, 7, 7], dtype=torch.float16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
