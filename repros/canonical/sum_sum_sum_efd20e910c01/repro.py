"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train
Pattern hash: efd20e910c01
Shape hash: 5624bcbd
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
_shapes_config = "(T([64, 4096, 92], f32), T([64, 2048, 92], f32), T([64, 2048, 372], f32), T([64, 1024, 372], f32), T([64, 1024, 1493], f32), T([64, 512, 1493], f32), T([64, 512, 5979], f32), T([64, 256, 5979], f32), T([64, 256, 23923], f32), T([64, 128, 23923], f32), T([64, 128, 95696], f32), T([64, 64, 95696], f32))"

class Repro(torch.nn.Module):
    def forward(self, cat: "f32[64, 4096, 92]", where: "f32[64, 2048, 92]", cat_1: "f32[64, 2048, 372]", where_1: "f32[64, 1024, 372]", cat_2: "f32[64, 1024, 1493]", where_2: "f32[64, 512, 1493]", cat_3: "f32[64, 512, 5979]", where_3: "f32[64, 256, 5979]", cat_4: "f32[64, 256, 23923]", where_4: "f32[64, 128, 23923]", cat_5: "f32[64, 128, 95696]", where_5: "f32[64, 64, 95696]"):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/demucs/demucs/model.py:215 in forward, code: x = encode(x)
        sum_dim_int_list: "f32[4096]" = torch.ops.aten.sum.dim_IntList(cat, [0, 2]);  cat = None
        sum_dim_int_list_1: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2]);  where = None
        sum_dim_int_list_2: "f32[2048]" = torch.ops.aten.sum.dim_IntList(cat_1, [0, 2]);  cat_1 = None
        sum_dim_int_list_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2]);  where_1 = None
        sum_dim_int_list_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(cat_2, [0, 2]);  cat_2 = None
        sum_dim_int_list_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2]);  where_2 = None
        sum_dim_int_list_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(cat_3, [0, 2]);  cat_3 = None
        sum_dim_int_list_7: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2]);  where_3 = None
        sum_dim_int_list_8: "f32[256]" = torch.ops.aten.sum.dim_IntList(cat_4, [0, 2]);  cat_4 = None
        sum_dim_int_list_9: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2]);  where_4 = None
        sum_dim_int_list_10: "f32[128]" = torch.ops.aten.sum.dim_IntList(cat_5, [0, 2]);  cat_5 = None
        sum_dim_int_list_11: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2]);  where_5 = None
        return (sum_dim_int_list, sum_dim_int_list_1, sum_dim_int_list_2, sum_dim_int_list_3, sum_dim_int_list_4, sum_dim_int_list_5, sum_dim_int_list_6, sum_dim_int_list_7, sum_dim_int_list_8, sum_dim_int_list_9, sum_dim_int_list_10, sum_dim_int_list_11)



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
