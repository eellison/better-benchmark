"""
Standalone repro captured via capture_hook.
Label: hf_AllenaiLongformerBase_train
Pattern hash: 72594f2ef62d
Shape hash: 796e385f
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
    def forward(self, mm_default: "f32[50268, 768]", view_7: "f32[8192, 50265]", _shape_param_0, view_9: "f32[8, 1024, 768]", mul_3: "f32[8, 1024, 768]", view_10: "f32[8192, 768]", _shape_param_1, mm_2: "f32[8192, 768]", _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1281 in forward, code: x = self.decoder(x)
        slice_tensor: "f32[50265, 768]" = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -3);  mm_default = None
        sum_dim_int_list: "f32[1, 50265]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True);  view_7 = None
        reshape_default: "f32[50265]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1278 in forward, code: x = self.layer_norm(x)
        mul_tensor: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_9, mul_3);  mul_3 = None
        sum_dim_int_list_1: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_9, [0, 1]);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1276 in forward, code: x = self.dense(features)
        permute_default: "f32[768, 8192]" = torch.ops.aten.permute.default(view_10, [1, 0])
        sum_dim_int_list_3: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_10, [0], True);  view_10 = None
        reshape_default_1: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None
        reshape_default_2: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_2, _shape_param_2);  mm_2 = _shape_param_2 = None
        return (slice_tensor, reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default, reshape_default_1, reshape_default_2)


def _default_make_inputs():
    return [
    torch.randn([50268, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 50265], dtype=torch.float32, device='cuda'),
    [50265],  # _shape_param_0
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [768],  # _shape_param_1
    torch.randn([8192, 768], dtype=torch.float32, device='cuda'),
    [8, 1024, 768],  # _shape_param_2
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
