"""
Standalone repro captured via capture_hook.
Label: torchbench_mobilenet_v3_large_train
Pattern hash: 6b84ffe7cd3b
Shape hash: f2681a3a
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
_shapes_config = "(T([256, 1280], f32))"

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[256, 1280]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:216 in _forward_impl, code: x = self.classifier(x)
        add_tensor: "f32[256, 1280]" = torch.ops.aten.add.Tensor(addmm, 3)
        clamp_min_default: "f32[256, 1280]" = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default: "f32[256, 1280]" = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor: "f32[256, 1280]" = torch.ops.aten.mul.Tensor(addmm, clamp_max_default);  addmm = clamp_max_default = None
        div_tensor: "f32[256, 1280]" = torch.ops.aten.div.Tensor(mul_tensor, 6);  mul_tensor = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv3.py:216 in _forward_impl, code: x = self.classifier(x)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[256, 1280]" = torch.ops.prims.inductor_random.default([256, 1280], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt_scalar: "b8[256, 1280]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.8);  inductor_random_default = None
        convert_element_type_default: "f32[256, 1280]" = torch.ops.prims.convert_element_type.default(lt_scalar, torch.float32);  lt_scalar = None
        div_scalar: "f32[256, 1280]" = torch.ops.aten.div.Scalar(convert_element_type_default, 0.8);  convert_element_type_default = None
        mul_tensor_1: "f32[256, 1280]" = torch.ops.aten.mul.Tensor(div_tensor, div_scalar);  div_tensor = div_scalar = None
        return mul_tensor_1



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
