"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: e9032453a6f7
Shape hash: 3f7fd6ec
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
_shapes_config = "(T([512, 512, 13, 13], b8, stride=(86528, 1, 6656, 512)), T([512, 512, 13, 13], f32, stride=(86528, 1, 6656, 512)), T([512, 256, 13, 13], b8, stride=(43264, 1, 3328, 256)), T([], f32), T([512, 256, 13, 13], b8, stride=(43264, 1, 3328, 256)))"

class Repro(torch.nn.Module):
    def forward(self, gt: "b8[512, 512, 13, 13]", getitem_6: "f32[512, 512, 13, 13]", le_1: "b8[512, 256, 13, 13]", full_default: "f32[]", le_2: "b8[512, 256, 13, 13]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        convert_element_type_default: "f32[512, 512, 13, 13]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 2.0);  convert_element_type_default = None
        mul_tensor_1: "f32[512, 512, 13, 13]" = torch.ops.aten.mul.Tensor(getitem_6, mul_tensor);  getitem_6 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_tensor: "f32[512, 256, 13, 13]" = torch.ops.aten.slice.Tensor(mul_tensor_1, 1, 0, 256)
        slice_tensor_1: "f32[512, 256, 13, 13]" = torch.ops.aten.slice.Tensor(mul_tensor_1, 1, 256, 512);  mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_self: "f32[512, 256, 13, 13]" = torch.ops.aten.where.self(le_1, full_default, slice_tensor_1);  le_1 = slice_tensor_1 = None
        where_self_1: "f32[512, 256, 13, 13]" = torch.ops.aten.where.self(le_2, full_default, slice_tensor);  le_2 = full_default = slice_tensor = None
        return (where_self_1, where_self)



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
