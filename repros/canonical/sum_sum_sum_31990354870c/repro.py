"""
Standalone repro captured via capture_hook.
Label: torchbench_squeezenet1_1_train
Pattern hash: 31990354870c
Shape hash: 78fe5b05
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
    def forward(self, where: "f32[512, 1000, 13, 13]", where_1: "f32[512, 256, 13, 13]", where_2: "f32[512, 256, 13, 13]", where_3: "f32[512, 64, 13, 13]", where_4: "f32[512, 256, 13, 13]", where_5: "f32[512, 256, 13, 13]", where_6: "f32[512, 64, 13, 13]", where_7: "f32[512, 192, 13, 13]", where_8: "f32[512, 192, 13, 13]", where_9: "f32[512, 48, 13, 13]", where_10: "f32[512, 192, 13, 13]", where_11: "f32[512, 192, 13, 13]", where_12: "f32[512, 48, 13, 13]", where_13: "f32[512, 128, 27, 27]", where_14: "f32[512, 128, 27, 27]", where_15: "f32[512, 32, 27, 27]", where_16: "f32[512, 128, 27, 27]", where_17: "f32[512, 128, 27, 27]", where_18: "f32[512, 32, 27, 27]", where_19: "f32[512, 64, 55, 55]", where_20: "f32[512, 64, 55, 55]", where_21: "f32[512, 16, 55, 55]", where_22: "f32[512, 64, 55, 55]", where_23: "f32[512, 64, 55, 55]", where_24: "f32[512, 16, 55, 55]", where_25: "f32[512, 64, 111, 111]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:96 in forward, code: x = self.classifier(x)
        sum_dim_int_list: "f32[1000]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3]);  where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        sum_dim_int_list_1: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3]);  where_1 = None
        sum_dim_int_list_2: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3]);  where_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        sum_dim_int_list_3: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3]);  where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        sum_dim_int_list_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3]);  where_4 = None
        sum_dim_int_list_5: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3]);  where_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        sum_dim_int_list_6: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3]);  where_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        sum_dim_int_list_7: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3]);  where_7 = None
        sum_dim_int_list_8: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3]);  where_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        sum_dim_int_list_9: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3]);  where_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        sum_dim_int_list_10: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3]);  where_10 = None
        sum_dim_int_list_11: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3]);  where_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        sum_dim_int_list_12: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3]);  where_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        sum_dim_int_list_13: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3]);  where_13 = None
        sum_dim_int_list_14: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3]);  where_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        sum_dim_int_list_15: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3]);  where_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        sum_dim_int_list_16: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3]);  where_16 = None
        sum_dim_int_list_17: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3]);  where_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        sum_dim_int_list_18: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3]);  where_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        sum_dim_int_list_19: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3]);  where_19 = None
        sum_dim_int_list_20: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3]);  where_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        sum_dim_int_list_21: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3]);  where_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:32 in forward, code: [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        sum_dim_int_list_22: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3]);  where_22 = None
        sum_dim_int_list_23: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3]);  where_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:30 in forward, code: x = self.squeeze_activation(self.squeeze(x))
        sum_dim_int_list_24: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3]);  where_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/squeezenet.py:95 in forward, code: x = self.features(x)
        sum_dim_int_list_25: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3]);  where_25 = None
        return (sum_dim_int_list, sum_dim_int_list_1, sum_dim_int_list_2, sum_dim_int_list_3, sum_dim_int_list_4, sum_dim_int_list_5, sum_dim_int_list_6, sum_dim_int_list_7, sum_dim_int_list_8, sum_dim_int_list_9, sum_dim_int_list_10, sum_dim_int_list_11, sum_dim_int_list_12, sum_dim_int_list_13, sum_dim_int_list_14, sum_dim_int_list_15, sum_dim_int_list_16, sum_dim_int_list_17, sum_dim_int_list_18, sum_dim_int_list_19, sum_dim_int_list_20, sum_dim_int_list_21, sum_dim_int_list_22, sum_dim_int_list_23, sum_dim_int_list_24, sum_dim_int_list_25)


def _default_make_inputs():
    return [
    torch.randn([512, 1000, 13, 13], dtype=torch.float32, device='cuda'),
    torch.randn(22151168, dtype=torch.float32, device='cuda').as_strided([512, 256, 13, 13], [43264, 1, 3328, 256]),  # where_1
    torch.randn(22151168, dtype=torch.float32, device='cuda').as_strided([512, 256, 13, 13], [43264, 1, 3328, 256]),  # where_2
    torch.randn(5537792, dtype=torch.float32, device='cuda').as_strided([512, 64, 13, 13], [10816, 1, 832, 64]),  # where_3
    torch.randn(22151168, dtype=torch.float32, device='cuda').as_strided([512, 256, 13, 13], [43264, 1, 3328, 256]),  # where_4
    torch.randn(22151168, dtype=torch.float32, device='cuda').as_strided([512, 256, 13, 13], [43264, 1, 3328, 256]),  # where_5
    torch.randn(5537792, dtype=torch.float32, device='cuda').as_strided([512, 64, 13, 13], [10816, 1, 832, 64]),  # where_6
    torch.randn(16613376, dtype=torch.float32, device='cuda').as_strided([512, 192, 13, 13], [32448, 1, 2496, 192]),  # where_7
    torch.randn(16613376, dtype=torch.float32, device='cuda').as_strided([512, 192, 13, 13], [32448, 1, 2496, 192]),  # where_8
    torch.randn(4153344, dtype=torch.float32, device='cuda').as_strided([512, 48, 13, 13], [8112, 1, 624, 48]),  # where_9
    torch.randn(16613376, dtype=torch.float32, device='cuda').as_strided([512, 192, 13, 13], [32448, 1, 2496, 192]),  # where_10
    torch.randn(16613376, dtype=torch.float32, device='cuda').as_strided([512, 192, 13, 13], [32448, 1, 2496, 192]),  # where_11
    torch.randn(4153344, dtype=torch.float32, device='cuda').as_strided([512, 48, 13, 13], [8112, 1, 624, 48]),  # where_12
    torch.randn(47775744, dtype=torch.float32, device='cuda').as_strided([512, 128, 27, 27], [93312, 1, 3456, 128]),  # where_13
    torch.randn(47775744, dtype=torch.float32, device='cuda').as_strided([512, 128, 27, 27], [93312, 1, 3456, 128]),  # where_14
    torch.randn(11943936, dtype=torch.float32, device='cuda').as_strided([512, 32, 27, 27], [23328, 1, 864, 32]),  # where_15
    torch.randn(47775744, dtype=torch.float32, device='cuda').as_strided([512, 128, 27, 27], [93312, 1, 3456, 128]),  # where_16
    torch.randn(47775744, dtype=torch.float32, device='cuda').as_strided([512, 128, 27, 27], [93312, 1, 3456, 128]),  # where_17
    torch.randn(11943936, dtype=torch.float32, device='cuda').as_strided([512, 32, 27, 27], [23328, 1, 864, 32]),  # where_18
    torch.randn(99123200, dtype=torch.float32, device='cuda').as_strided([512, 64, 55, 55], [193600, 1, 3520, 64]),  # where_19
    torch.randn(99123200, dtype=torch.float32, device='cuda').as_strided([512, 64, 55, 55], [193600, 1, 3520, 64]),  # where_20
    torch.randn(24780800, dtype=torch.float32, device='cuda').as_strided([512, 16, 55, 55], [48400, 1, 880, 16]),  # where_21
    torch.randn(99123200, dtype=torch.float32, device='cuda').as_strided([512, 64, 55, 55], [193600, 1, 3520, 64]),  # where_22
    torch.randn(99123200, dtype=torch.float32, device='cuda').as_strided([512, 64, 55, 55], [193600, 1, 3520, 64]),  # where_23
    torch.randn(24780800, dtype=torch.float32, device='cuda').as_strided([512, 16, 55, 55], [48400, 1, 880, 16]),  # where_24
    torch.randn(403734528, dtype=torch.float32, device='cuda').as_strided([512, 64, 111, 111], [788544, 1, 7104, 64]),  # where_25
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
