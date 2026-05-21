"""
Standalone repro captured via capture_hook.
Label: torchbench_alexnet_train
Pattern hash: 9dd701a7b239
Shape hash: 0d988dfc
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
_shapes_config = "(T([1024, 1000], f32), T([1024, 4096], f32), T([1024, 4096], f32), T([1024, 256, 13, 13], f32), T([1024, 256, 13, 13], f32, stride=(43264, 1, 3328, 256)), T([1024, 384, 13, 13], f32, stride=(64896, 1, 4992, 384)), T([1024, 192, 27, 27], f32, stride=(139968, 1, 5184, 192)), T([1024, 64, 55, 55], f32, stride=(193600, 1, 3520, 64)), S([1000]), S([4096]), S([4096]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[1024, 1000]", where: "f32[1024, 4096]", where_1: "f32[1024, 4096]", where_2: "f32[1024, 256, 13, 13]", where_3: "f32[1024, 256, 13, 13]", where_4: "f32[1024, 384, 13, 13]", where_5: "f32[1024, 192, 27, 27]", where_6: "f32[1024, 64, 55, 55]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:51 in forward, code: x = self.classifier(x)
        permute_default: "f32[1000, 1024]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        permute_default_1: "f32[4096, 1024]" = torch.ops.aten.permute.default(where, [1, 0])
        sum_dim_int_list_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        permute_default_2: "f32[4096, 1024]" = torch.ops.aten.permute.default(where_1, [1, 0])
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        reshape_default_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_2);  sum_dim_int_list_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/alexnet.py:48 in forward, code: x = self.features(x)
        sum_dim_int_list_3: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3]);  where_2 = None
        sum_dim_int_list_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3]);  where_3 = None
        sum_dim_int_list_5: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3]);  where_4 = None
        sum_dim_int_list_6: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3]);  where_5 = None
        sum_dim_int_list_7: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3]);  where_6 = None
        return (permute_default, reshape_default, permute_default_1, reshape_default_1, permute_default_2, reshape_default_2, sum_dim_int_list_3, sum_dim_int_list_4, sum_dim_int_list_5, sum_dim_int_list_6, sum_dim_int_list_7)



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
