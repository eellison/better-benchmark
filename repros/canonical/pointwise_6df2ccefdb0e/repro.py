"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: 6df2ccefdb0e
Shape hash: fdfcb0af
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
_shapes_config = "(T([512, 256, 13, 13], f32, stride=(43264, 1, 3328, 256)), T([512, 256, 13, 13], i8, stride=(43264, 1, 3328, 256), gen=Index(9)), T([512, 128, 27, 27], b8, stride=(93312, 1, 3456, 128)), T([], f32), T([512, 128, 27, 27], b8, stride=(93312, 1, 3456, 128)), S([131072, 169]), S([131072, 169]), S([512, 256, 27, 27]))"

class Repro(torch.nn.Module):
    def forward(self, getitem_42: "f32[512, 256, 13, 13]", getitem_5: "i8[512, 256, 13, 13]", le_13: "b8[512, 128, 27, 27]", full_default: "f32[]", le_14: "b8[512, 128, 27, 27]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        full_default_1: "f32[131072, 729]" = torch.ops.aten.full.default([131072, 729], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        reshape_default: "f32[131072, 169]" = torch.ops.aten.reshape.default(getitem_42, _shape_param_0);  getitem_42 = _shape_param_0 = None
        _low_memory_max_pool_offsets_to_indices_default: "i64[512, 256, 13, 13]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_5, [3, 3], [27, 27], [2, 2], [0, 0], [1, 1]);  getitem_5 = None
        reshape_default_1: "i64[131072, 169]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_default, _shape_param_1);  _low_memory_max_pool_offsets_to_indices_default = _shape_param_1 = None
        scatter_add_default: "f32[131072, 729]" = torch.ops.aten.scatter_add.default(full_default_1, 1, reshape_default_1, reshape_default);  full_default_1 = reshape_default_1 = reshape_default = None
        reshape_default_2: "f32[512, 256, 27, 27]" = torch.ops.aten.reshape.default(scatter_add_default, _shape_param_2);  scatter_add_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:31 in forward, code: return torch.cat(
        slice_tensor: "f32[512, 128, 27, 27]" = torch.ops.aten.slice.Tensor(reshape_default_2, 1, 0, 128)
        slice_tensor_1: "f32[512, 128, 27, 27]" = torch.ops.aten.slice.Tensor(reshape_default_2, 1, 128, 256);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        where_self: "f32[512, 128, 27, 27]" = torch.ops.aten.where.self(le_13, full_default, slice_tensor_1);  le_13 = slice_tensor_1 = None
        where_self_1: "f32[512, 128, 27, 27]" = torch.ops.aten.where.self(le_14, full_default, slice_tensor);  le_14 = full_default = slice_tensor = None
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
