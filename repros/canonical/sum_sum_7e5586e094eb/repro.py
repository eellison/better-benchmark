"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-2-5-linux.aws.a100_graph5
Pattern hash: 7e5586e094eb
Shape hash: c3773531
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
    def forward(self, mm_6: "f32[512, 768]", mul_26: "f32[1, 512, 768]", arg50_1: "f32[768]", arg126_1: "f32[1, 512, 768]", arg142_1: "f32[1, 512, 1]", _shape_param_0):
        # No stacktrace found for following nodes
        view_default: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_6, _shape_param_0);  mm_6 = _shape_param_0 = None
        add_tensor: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_26, view_default);  mul_26 = view_default = None
        mul_tensor: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_tensor, arg50_1);  add_tensor = arg50_1 = None
        mul_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True)
        mul_tensor_2: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, arg126_1);  mul_tensor = None
        sum_dim_int_list_1: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True);  mul_tensor_2 = None
        mul_tensor_3: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg126_1, sum_dim_int_list_1);  arg126_1 = sum_dim_int_list_1 = None
        sub_tensor: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(arg142_1, sub_tensor_1);  arg142_1 = sub_tensor_1 = None
        full_default: "f32[1, 512, 768, 2]" = torch.ops.aten.full.default([1, 512, 768, 2], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[1, 512, 768, 2]" = torch.ops.aten.select_scatter.default(full_default, mul_tensor_4, 3, 0);  full_default = mul_tensor_4 = None
        return select_scatter_default


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1], dtype=torch.float32, device='cuda'),
    [1, 512, 768],  # _shape_param_0
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
