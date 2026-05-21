"""
Standalone repro captured via capture_hook.
Label: torchbench_vgg16_train
Pattern hash: e70c66b80471
Shape hash: cdfc0c15
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
    def forward(self, tangents_1: "f32[128, 1000]", where: "f32[128, 4096]", where_1: "f32[128, 4096]", where_2: "f32[128, 512, 14, 14]", where_3: "f32[128, 512, 14, 14]", where_4: "f32[128, 512, 14, 14]", where_5: "f32[128, 512, 28, 28]", where_6: "f32[128, 512, 28, 28]", where_7: "f32[128, 512, 28, 28]", where_8: "f32[128, 256, 56, 56]", where_9: "f32[128, 256, 56, 56]", where_10: "f32[128, 256, 56, 56]", where_11: "f32[128, 128, 112, 112]", where_12: "f32[128, 128, 112, 112]", where_13: "f32[128, 64, 224, 224]", where_14: "f32[128, 64, 224, 224]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:69 in forward, code: x = self.classifier(x)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None
        permute_default_1: "f32[4096, 128]" = torch.ops.aten.permute.default(where, [1, 0])
        sum_dim_int_list_1: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where, [0], True);  where = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None
        permute_default_2: "f32[4096, 128]" = torch.ops.aten.permute.default(where_1, [1, 0])
        sum_dim_int_list_2: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
        reshape_default_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_2);  sum_dim_int_list_2 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/vgg.py:66 in forward, code: x = self.features(x)
        sum_dim_int_list_3: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3]);  where_2 = None
        sum_dim_int_list_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3]);  where_3 = None
        sum_dim_int_list_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3]);  where_4 = None
        sum_dim_int_list_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3]);  where_5 = None
        sum_dim_int_list_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3]);  where_6 = None
        sum_dim_int_list_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3]);  where_7 = None
        sum_dim_int_list_9: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3]);  where_8 = None
        sum_dim_int_list_10: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3]);  where_9 = None
        sum_dim_int_list_11: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3]);  where_10 = None
        sum_dim_int_list_12: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3]);  where_11 = None
        sum_dim_int_list_13: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3]);  where_12 = None
        sum_dim_int_list_14: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3]);  where_13 = None
        sum_dim_int_list_15: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3]);  where_14 = None
        return (permute_default, reshape_default, permute_default_1, reshape_default_1, permute_default_2, reshape_default_2, sum_dim_int_list_3, sum_dim_int_list_4, sum_dim_int_list_5, sum_dim_int_list_6, sum_dim_int_list_7, sum_dim_int_list_8, sum_dim_int_list_9, sum_dim_int_list_10, sum_dim_int_list_11, sum_dim_int_list_12, sum_dim_int_list_13, sum_dim_int_list_14, sum_dim_int_list_15)


def _default_make_inputs():
    return [
    torch.randn([128, 1000], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 4096], dtype=torch.float32, device='cuda'),
    torch.randn([128, 512, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn(12845056, dtype=torch.float32, device='cuda').as_strided([128, 512, 14, 14], [100352, 1, 7168, 512]),  # where_3
    torch.randn(12845056, dtype=torch.float32, device='cuda').as_strided([128, 512, 14, 14], [100352, 1, 7168, 512]),  # where_4
    torch.randn(51380224, dtype=torch.float32, device='cuda').as_strided([128, 512, 28, 28], [401408, 1, 14336, 512]),  # where_5
    torch.randn(51380224, dtype=torch.float32, device='cuda').as_strided([128, 512, 28, 28], [401408, 1, 14336, 512]),  # where_6
    torch.randn(51380224, dtype=torch.float32, device='cuda').as_strided([128, 512, 28, 28], [401408, 1, 14336, 512]),  # where_7
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([128, 256, 56, 56], [802816, 1, 14336, 256]),  # where_8
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([128, 256, 56, 56], [802816, 1, 14336, 256]),  # where_9
    torch.randn(102760448, dtype=torch.float32, device='cuda').as_strided([128, 256, 56, 56], [802816, 1, 14336, 256]),  # where_10
    torch.randn(205520896, dtype=torch.float32, device='cuda').as_strided([128, 128, 112, 112], [1605632, 1, 14336, 128]),  # where_11
    torch.randn(205520896, dtype=torch.float32, device='cuda').as_strided([128, 128, 112, 112], [1605632, 1, 14336, 128]),  # where_12
    torch.randn(411041792, dtype=torch.float32, device='cuda').as_strided([128, 64, 224, 224], [3211264, 1, 14336, 64]),  # where_13
    torch.randn(411041792, dtype=torch.float32, device='cuda').as_strided([128, 64, 224, 224], [3211264, 1, 14336, 64]),  # where_14
    [1000],  # _shape_param_0
    [4096],  # _shape_param_1
    [4096],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
