"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 0a6d7200d2ab
Shape hash: 2aded74a
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
_shapes_config = "(T([1, 192, 1, 1], f32), T([192], f32), T([128, 192, 28, 28], f32, stride=(150528, 1, 5376, 192)))"

class Repro(torch.nn.Module):
    def forward(self, getitem_7: "f32[1, 192, 1, 1]", primals_25: "f32[192]", add_18: "f32[128, 192, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        squeeze_dims: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        mul_tensor: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_1: "f32[192]" = torch.ops.aten.mul.Tensor(primals_25, 0.9)
        add_tensor: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        unsqueeze_default: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_1: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        sub_tensor: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_18, unsqueeze_default_2);  add_18 = unsqueeze_default_2 = None

        # No stacktrace found for following nodes
        copy__default: "f32[192]" = torch.ops.aten.copy_.default(primals_25, add_tensor);  primals_25 = add_tensor = None
        return (copy__default, sub_tensor)



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
